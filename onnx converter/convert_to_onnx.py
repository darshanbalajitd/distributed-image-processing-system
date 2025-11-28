import os
import sys

# Make sure we can import model.py and resnet.py from this folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import keras

from model import Vggface2_ResNet50
import resnet  # just to ensure it's registered

WEIGHTS_PATH = "weights.h5"
OUTPUT_ONNX = "vggface2_resnet50.onnx"


def main():
    print("Building training model (with classifier)...")

    # Build model in TRAIN mode so its architecture matches the saved weights
    base_model = Vggface2_ResNet50(mode="train")

    print("Loading weights from:", WEIGHTS_PATH)
    # Let Keras match by layer name; skip the mismatched extra layer if needed
    base_model.load_weights(WEIGHTS_PATH)

    print("Creating feature-extractor model (512-dim L2-normalized)...")

    # Get the 512-dim dense layer before classifier
    dim_proj_layer = base_model.get_layer("dim_proj")
    embeddings = dim_proj_layer.output

    # Add an explicit L2-normalization layer on top
    embeddings = keras.layers.Lambda(
        lambda x: keras.backend.l2_normalize(x, axis=1),
        output_shape=(512,),
        name="l2_normalize"
    )(embeddings)

    feature_model = keras.models.Model(
        inputs=base_model.input,
        outputs=embeddings,
        name="vggface2_resnet50_feature"
    )

    print("Exporting feature model to ONNX...")

    try:
        import tf2onnx
    except ImportError:
        print("ERROR: tf2onnx is not installed. Please run:")
        print("    pip install tf2onnx")
        return

    # Input: batch of RGB images 224x224x3
    spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        feature_model,
        input_signature=spec,
        opset=13,
        output_path=OUTPUT_ONNX,
    )

    print(f"✅ Successfully saved ONNX model to: {OUTPUT_ONNX}")


if __name__ == "__main__":
    main()
