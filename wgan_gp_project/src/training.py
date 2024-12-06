import tensorflow as tf
from src.models import build_critic, build_generator, gradient_penalty
import matplotlib.pyplot as plt
import numpy as np

# Training step
@tf.function
def train_step(real_data, generator, critic, gen_opt, crit_opt, lambda_gp=10):
    batch_size = tf.shape(real_data)[0]

    # Train Critic
    for _ in range(5):
        noise = tf.random.normal([batch_size, 100])
        with tf.GradientTape() as tape:
            fake_data = generator(noise, training=True)
            real_output = critic(real_data, training=True)
            fake_output = critic(fake_data, training=True)
            gp_loss = gradient_penalty(critic, real_data, fake_data, lambda_gp)
            d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + gp_loss

        gradients = tape.gradient(d_loss, critic.trainable_variables)
        crit_opt.apply_gradients(zip(gradients, critic.trainable_variables))

    # Train Generator
    noise = tf.random.normal([batch_size, 100])
    with tf.GradientTape() as tape:
        fake_data = generator(noise, training=True)
        fake_output = critic(fake_data, training=True)
        g_loss = -tf.reduce_mean(fake_output)

    gradients = tape.gradient(g_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gradients, generator.trainable_variables))

    return d_loss, g_loss

# Training function
def train_wgan_gp(data, epochs, batch_size, model_name):
    generator = build_generator()
    critic = build_critic()

    gen_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
    crit_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    d_losses, g_losses = [], []

    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx].astype(np.float32)

        d_loss, g_loss = train_step(real_data, generator, critic, gen_opt, crit_opt)
        d_losses.append(d_loss.numpy())
        g_losses.append(g_loss.numpy())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss = {d_loss.numpy()}, G Loss = {g_loss.numpy()}")

    plt.plot(d_losses, label="Critic Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.legend()
    plt.show()

    return generator