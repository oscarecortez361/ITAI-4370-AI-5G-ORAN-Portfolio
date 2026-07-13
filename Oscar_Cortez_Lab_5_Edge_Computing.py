"""
Laboratory 05: Edge Computing Simulation with a Lightweight Framework
Student: Oscar Cortez
Course: ITAI-4370
"""

# ============================================================================
# Core libraries
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

# Reproducibility
SEED = 42
keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# Use CPU only to simulate an edge device.
try:
    tf.config.set_visible_devices([], "GPU")
    print("GPU disabled. Running the simulation on CPU.")
except (RuntimeError, ValueError):
    print("TensorFlow was already initialized; continuing with the available CPU configuration.")

# Limit thread use to better represent a constrained edge environment.
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except RuntimeError:
    pass

# ============================================================================
print("\n" + "=" * 72)
print("GENERATING IOT SENSOR DATASET")
print("=" * 72)

X, y = make_classification(
    n_samples=10_000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    class_sep=1.25,
    random_state=SEED
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

NUM_FEATURES = X_train.shape[1]
NUM_CLASSES = len(np.unique(y_train))

print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples:     {X_test.shape[0]:,}")
print(f"Features:         {NUM_FEATURES}")
print(f"Classes:          {NUM_CLASSES}")
print("Training class counts:", np.bincount(y_train))
print("Test class counts:    ", np.bincount(y_test))

# ============================================================================
def create_baseline_model(input_dim, num_classes):
    """Create the full-precision baseline model."""
    return keras.Sequential(
        [
            layers.Input(shape=(input_dim,), name="sensor_input"),
            layers.Dense(128, activation="relu", name="dense_128"),
            layers.Dropout(0.30),
            layers.Dense(64, activation="relu", name="dense_64"),
            layers.Dropout(0.30),
            layers.Dense(32, activation="relu", name="dense_32"),
            layers.Dense(num_classes, name="class_logits")
        ],
        name="baseline_cloud_model"
    )


def compile_classifier(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return model


baseline_model = compile_classifier(
    create_baseline_model(NUM_FEATURES, NUM_CLASSES)
)

print("\nBaseline model architecture:")
baseline_model.summary()

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

baseline_history = baseline_model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

baseline_loss, baseline_accuracy = baseline_model.evaluate(
    X_test, y_test, verbose=0
)

print(f"\nBaseline test loss:     {baseline_loss:.4f}")
print(f"Baseline test accuracy: {baseline_accuracy:.4f}")
print(f"Baseline parameters:    {baseline_model.count_params():,}")

# ============================================================================
def create_structured_pruned_model(
    input_dim,
    num_classes,
    prune_fraction=0.50
):
    """Simulate structured pruning by reducing hidden-layer widths."""
    if not 0.0 <= prune_fraction < 1.0:
        raise ValueError("prune_fraction must be between 0.0 and 1.0.")

    original_units = [128, 64, 32]
    pruned_units = [
        max(4, int(round(units * (1.0 - prune_fraction))))
        for units in original_units
    ]

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,), name="sensor_input"),
            layers.Dense(pruned_units[0], activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(pruned_units[1], activation="relu"),
            layers.Dense(pruned_units[2], activation="relu"),
            layers.Dense(num_classes, name="class_logits")
        ],
        name="structured_pruned_model"
    )
    return model, pruned_units


pruned_model, pruned_units = create_structured_pruned_model(
    NUM_FEATURES,
    NUM_CLASSES,
    prune_fraction=0.50
)
pruned_model = compile_classifier(pruned_model)

pruned_history = pruned_model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=20,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        )
    ],
    verbose=0
)

pruned_loss, pruned_accuracy = pruned_model.evaluate(
    X_test, y_test, verbose=0
)

parameter_reduction = (
    1.0 - pruned_model.count_params() / baseline_model.count_params()
) * 100

print("Pruned hidden-layer widths:", pruned_units)
print(f"Pruned test accuracy:       {pruned_accuracy:.4f}")
print(f"Pruned parameters:          {pruned_model.count_params():,}")
print(f"Parameter reduction:        {parameter_reduction:.2f}%")

# ============================================================================
def create_student_model(input_dim, num_classes):
    """Create the smaller student network."""
    return keras.Sequential(
        [
            layers.Input(shape=(input_dim,), name="sensor_input"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, name="class_logits")
        ],
        name="distilled_student_model"
    )


class Distiller(keras.Model):
    """Train a student model from hard labels and teacher soft targets."""

    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = keras.metrics.Mean(
            name="distillation_loss"
        )
        self.accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.student_loss_tracker,
            self.distillation_loss_tracker,
            self.accuracy_tracker
        ]

    def compile(
        self,
        optimizer,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.50,
        temperature=3.0
    ):
        super().compile(optimizer=optimizer)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data
        teacher_logits = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_logits)

            teacher_soft = tf.nn.softmax(
                teacher_logits / self.temperature,
                axis=1
            )
            student_soft = tf.nn.softmax(
                student_logits / self.temperature,
                axis=1
            )

            distillation_loss = self.distillation_loss_fn(
                teacher_soft,
                student_soft
            ) * (self.temperature ** 2)

            total_loss = (
                self.alpha * student_loss
                + (1.0 - self.alpha) * distillation_loss
            )

        gradients = tape.gradient(
            total_loss,
            self.student.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(gradients, self.student.trainable_variables)
        )

        self.total_loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.accuracy_tracker.update_state(y, student_logits)

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        x, y = data
        student_logits = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, student_logits)

        self.total_loss_tracker.update_state(student_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(0.0)
        self.accuracy_tracker.update_state(y, student_logits)

        return {metric.name: metric.result() for metric in self.metrics}


student_model = create_student_model(NUM_FEATURES, NUM_CLASSES)

distiller = Distiller(
    student=student_model,
    teacher=baseline_model
)

distiller.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.50,
    temperature=3.0
)

distillation_history = distiller.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=20,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        )
    ],
    verbose=0
)

# Compile the trained student normally for final evaluation and conversion.
student_model = compile_classifier(student_model)

student_loss, distilled_accuracy = student_model.evaluate(
    X_test, y_test, verbose=0
)

print(f"Distilled student accuracy: {distilled_accuracy:.4f}")
print(f"Distilled parameters:       {student_model.count_params():,}")
print(
    "Parameter reduction:        "
    f"{(1 - student_model.count_params() / baseline_model.count_params()) * 100:.2f}%"
)

# ============================================================================
def convert_float_tflite(model):
    """Convert a Keras model to float32 TensorFlow Lite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def representative_dataset():
    """Provide representative input samples for INT8 calibration."""
    sample_count = min(250, len(X_train))
    for index in range(sample_count):
        yield [X_train[index:index + 1].astype(np.float32)]


def convert_int8_tflite(model):
    """Convert a Keras model to full-integer INT8 TensorFlow Lite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


baseline_tflite = convert_float_tflite(baseline_model)
pruned_tflite = convert_float_tflite(pruned_model)
distilled_tflite = convert_float_tflite(student_model)
quantized_tflite = convert_int8_tflite(baseline_model)

Path("quantized_model_int8.tflite").write_bytes(quantized_tflite)

print(f"Baseline float TFLite size:  {len(baseline_tflite) / 1024:.2f} KB")
print(f"Pruned float TFLite size:    {len(pruned_tflite) / 1024:.2f} KB")
print(f"Distilled float TFLite size: {len(distilled_tflite) / 1024:.2f} KB")
print(f"Quantized INT8 size:         {len(quantized_tflite) / 1024:.2f} KB")
print("Saved: quantized_model_int8.tflite")

# ============================================================================
def prepare_tflite_input(sample, input_details):
    """Convert one sample to the TensorFlow Lite input dtype."""
    dtype = input_details["dtype"]
    sample = sample.reshape(1, -1)

    if np.issubdtype(dtype, np.integer):
        scale, zero_point = input_details["quantization"]
        if scale == 0:
            raise ValueError("The quantized input scale cannot be zero.")

        quantized = np.round(sample / scale + zero_point)
        limits = np.iinfo(dtype)
        quantized = np.clip(quantized, limits.min, limits.max)
        return quantized.astype(dtype)

    return sample.astype(dtype)


def dequantize_tflite_output(output, output_details):
    """Convert a quantized output tensor back to float values."""
    dtype = output_details["dtype"]

    if np.issubdtype(dtype, np.integer):
        scale, zero_point = output_details["quantization"]
        return (output.astype(np.float32) - zero_point) * scale

    return output.astype(np.float32)


def predict_tflite(model_content, data):
    """Return class predictions for a TensorFlow Lite model."""
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []

    for sample in data:
        prepared = prepare_tflite_input(sample, input_details)
        interpreter.set_tensor(input_details["index"], prepared)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details["index"])
        output = dequantize_tflite_output(output, output_details)
        predictions.append(int(np.argmax(output[0])))

    return np.asarray(predictions, dtype=np.int32)


def measure_tflite_latency(
    model_content,
    sample,
    warmup_runs=20,
    timed_runs=200
):
    """Measure average single-sample TensorFlow Lite latency in ms."""
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    prepared = prepare_tflite_input(sample, input_details)

    for _ in range(warmup_runs):
        interpreter.set_tensor(input_details["index"], prepared)
        interpreter.invoke()

    timings = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], prepared)
        interpreter.invoke()
        timings.append(time.perf_counter() - start)

    return float(np.mean(timings) * 1000.0)


deployment_models = {
    "Baseline Float32": {
        "bytes": baseline_tflite,
        "parameters": baseline_model.count_params()
    },
    "Structured Pruned": {
        "bytes": pruned_tflite,
        "parameters": pruned_model.count_params()
    },
    "Distilled Student": {
        "bytes": distilled_tflite,
        "parameters": student_model.count_params()
    },
    "Quantized INT8": {
        "bytes": quantized_tflite,
        "parameters": baseline_model.count_params()
    }
}

prediction_store = {}
results = []

for model_name, model_info in deployment_models.items():
    model_bytes = model_info["bytes"]

    predictions = predict_tflite(model_bytes, X_test)
    prediction_store[model_name] = predictions

    accuracy = accuracy_score(y_test, predictions)
    latency_ms = measure_tflite_latency(
        model_bytes,
        X_test[0],
        warmup_runs=20,
        timed_runs=200
    )
    size_kb = len(model_bytes) / 1024.0

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Inference Time (ms)": latency_ms,
            "Size (KB)": size_kb,
            "Parameters": model_info["parameters"]
        }
    )

results_df = pd.DataFrame(results)

baseline_size = float(
    results_df.loc[
        results_df["Model"] == "Baseline Float32",
        "Size (KB)"
    ].iloc[0]
)

results_df["Compression Ratio"] = (
    baseline_size / results_df["Size (KB)"]
)
results_df["Accuracy Loss vs Baseline"] = (
    results_df.loc[
        results_df["Model"] == "Baseline Float32",
        "Accuracy"
    ].iloc[0]
    - results_df["Accuracy"]
)

results_df.to_csv("edge_model_results.csv", index=False)

print("\n" + "=" * 72)
print("EDGE INFERENCE RESULTS")
print("=" * 72)
print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Inference Time (ms)": "{:.4f}".format,
            "Size (KB)": "{:.2f}".format,
            "Compression Ratio": "{:.2f}".format,
            "Accuracy Loss vs Baseline": "{:.4f}".format
        }
    )
)
print("\nSaved: edge_model_results.csv")

# ============================================================================
for model_name in ["Baseline Float32", "Quantized INT8"]:
    print("\n" + "=" * 72)
    print(model_name.upper())
    print("=" * 72)
    print(
        classification_report(
            y_test,
            prediction_store[model_name],
            digits=4,
            zero_division=0
        )
    )

# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Accuracy
axes[0, 0].bar(results_df["Model"], results_df["Accuracy"])
axes[0, 0].set_title("Model Accuracy Comparison")
axes[0, 0].set_ylabel("Accuracy")
lower_limit = max(0.0, results_df["Accuracy"].min() - 0.05)
axes[0, 0].set_ylim(lower_limit, 1.0)
axes[0, 0].grid(axis="y", alpha=0.3)

for index, value in enumerate(results_df["Accuracy"]):
    axes[0, 0].text(index, value, f"{value:.3f}", ha="center", va="bottom")

# Model size
axes[0, 1].bar(results_df["Model"], results_df["Size (KB)"])
axes[0, 1].set_title("TensorFlow Lite Model Size")
axes[0, 1].set_ylabel("Size (KB)")
axes[0, 1].grid(axis="y", alpha=0.3)

for index, value in enumerate(results_df["Size (KB)"]):
    axes[0, 1].text(index, value, f"{value:.1f}", ha="center", va="bottom")

# Latency
axes[1, 0].bar(
    results_df["Model"],
    results_df["Inference Time (ms)"]
)
axes[1, 0].set_title("Measured Single-Sample Inference Time")
axes[1, 0].set_ylabel("Milliseconds")
axes[1, 0].grid(axis="y", alpha=0.3)

for index, value in enumerate(results_df["Inference Time (ms)"]):
    axes[1, 0].text(index, value, f"{value:.3f}", ha="center", va="bottom")

# Compression versus accuracy
axes[1, 1].scatter(
    results_df["Compression Ratio"],
    results_df["Accuracy"],
    s=180,
    alpha=0.75
)
axes[1, 1].set_title("Compression vs. Accuracy Tradeoff")
axes[1, 1].set_xlabel("Compression Ratio")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].grid(alpha=0.3)

for _, row in results_df.iterrows():
    axes[1, 1].annotate(
        row["Model"],
        (row["Compression Ratio"], row["Accuracy"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9
    )

for axis in axes.flat:
    axis.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig(
    "edge_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("Saved: edge_model_comparison.png")

# ============================================================================
baseline_row = results_df[
    results_df["Model"] == "Baseline Float32"
].iloc[0]

quantized_row = results_df[
    results_df["Model"] == "Quantized INT8"
].iloc[0]

compact_candidates = results_df[
    results_df["Model"].isin(
        ["Structured Pruned", "Distilled Student"]
    )
].copy()

# Prefer a compact model within two percentage points of baseline accuracy.
eligible = compact_candidates[
    compact_candidates["Accuracy"]
    >= baseline_row["Accuracy"] - 0.02
]

if eligible.empty:
    gateway_choice = compact_candidates.sort_values(
        ["Accuracy", "Size (KB)"],
        ascending=[False, True]
    ).iloc[0]
else:
    gateway_choice = eligible.sort_values(
        ["Size (KB)", "Inference Time (ms)"],
        ascending=[True, True]
    ).iloc[0]

print("\n" + "=" * 72)
print("EDGE DEPLOYMENT RECOMMENDATIONS")
print("=" * 72)

print("\nMicrocontroller / wearable")
print("Recommended model: Quantized INT8")
print(
    f"Reason: {quantized_row['Size (KB)']:.2f} KB deployed size, "
    f"{quantized_row['Accuracy']:.4f} accuracy, and measured INT8 inference."
)

print("\nEdge gateway")
print(f"Recommended model: {gateway_choice['Model']}")
print(
    f"Reason: {gateway_choice['Size (KB)']:.2f} KB deployed size with "
    f"{gateway_choice['Accuracy']:.4f} accuracy."
)

print("\nEdge server")
print("Recommended model: Baseline Float32")
print(
    f"Reason: highest-capacity model with {baseline_row['Accuracy']:.4f} "
    "accuracy when memory and processing resources are available."
)

# ============================================================================
smallest_row = results_df.sort_values("Size (KB)").iloc[0]
fastest_row = results_df.sort_values("Inference Time (ms)").iloc[0]
most_accurate_row = results_df.sort_values(
    "Accuracy",
    ascending=False
).iloc[0]

print("LAB 5 CONCLUSION")
print("-" * 72)
print(
    f"The baseline model achieved {baseline_row['Accuracy']:.4f} accuracy "
    f"with a deployed size of {baseline_row['Size (KB)']:.2f} KB."
)
print(
    f"The smallest model was {smallest_row['Model']} at "
    f"{smallest_row['Size (KB)']:.2f} KB, giving a "
    f"{smallest_row['Compression Ratio']:.2f}x compression ratio."
)
print(
    f"The fastest measured model was {fastest_row['Model']} at "
    f"{fastest_row['Inference Time (ms)']:.4f} ms per sample."
)
print(
    f"The most accurate deployed model was {most_accurate_row['Model']} "
    f"with {most_accurate_row['Accuracy']:.4f} accuracy."
)
print(
    "\nOverall, the experiment demonstrates the main edge-AI tradeoff: "
    "model optimization reduces memory and can reduce latency, but the "
    "accuracy must still be checked before deployment. INT8 quantization "
    "is the strongest choice for highly constrained devices, while a "
    "structured-pruned or distilled model can provide a useful balance "
    "for an edge gateway. The baseline remains appropriate when an edge "
    "server has enough resources and accuracy is the highest priority."
)
