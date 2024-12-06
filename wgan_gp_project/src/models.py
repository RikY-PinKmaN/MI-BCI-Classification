import tensorflow as tf
from tensorflow.keras import layers

# Generator
def build_generator():
    noise_input = layers.Input(shape=(100,))  # Noise vector (latent space)

    # Fully connected layer to project noise into a larger space
    x = layers.Dense(64 * 63)(noise_input)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Reshape((63, 64))(x)  # Shape: (63, 64)

    # Upsampling layers with Conv1DTranspose
    x = layers.Conv1DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False)(x)  # Shape: (126, 128)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False)(x)  # Shape: (252, 64)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1DTranspose(22, kernel_size=7, strides=2, padding="same", activation="tanh")(x)  # Shape: (504, 22)

    # Crop the extra time points to match (22, 500)
    x = layers.Cropping1D(cropping=(2, 2))(x)  # Shape: (500, 22)

    # Permute to match critic's expected input shape
    x = layers.Permute((2, 1))(x)  # Shape: (22, 500)

    return tf.keras.Model(noise_input, x)

# Improved Critic
def build_critic():
    data_input = layers.Input(shape=(22, 500))
    x = layers.Permute((2, 1))(data_input)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1D(128, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1D(256, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv1D(512, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return tf.keras.Model(data_input, x)

# Gradient penalty
def gradient_penalty(critic, real_samples, fake_samples, lambda_gp=10):
    alpha = tf.random.uniform(shape=[tf.shape(real_samples)[0], 1, 1], minval=0., maxval=1.)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        interpolated_predictions = critic(interpolated, training=True)

    gradients = gp_tape.gradient(interpolated_predictions, [interpolated])[0]
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]) + 1e-10)
    gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)
    return lambda_gp * gradient_penalty