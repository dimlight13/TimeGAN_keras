import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Activation, Concatenate, GRU, Embedding
from tqdm import tqdm
from tensorflow.keras import Model
from tensorflow import function, GradientTape, sqrt, abs, reduce_mean, ones_like, zeros_like, convert_to_tensor
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow import data as tfdata
from tensorflow.keras.optimizers import Adam
from tensorflow import nn
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras import layers

PATH  = "/content/gdrive/MyDrive/GAN_model_save/"

class TSGAN:
    def __init__(self, dataset, label, is_train=False):
        self.label = label
        self.n_classes = np.unique(label).shape[0]
        self.is_train = is_train
        self.data = dataset
        self.latent_dim = 512
        self.seq_len = self.data.shape[1]
        self.n_seq = self.data.shape[-1]
        self.gamma = 1
        self.hidden_dim = 128
        self.d_model = self.build_discriminator()
        self.g_model = self.build_generator()
        self.valid = None
        if self.is_train == True:
            self.supervisor = load_model(PATH + "supervisor.h5")
            self.recovery = load_model(PATH + "rec_layer.h5")
            self.embedder = load_model(PATH + "emb_layer.h5")
            H = self.embedder.output
            X_tilde = self.recovery(H)
            self.autoencoder = Model(inputs=self.embedder.input, outputs=X_tilde)
        else:
            self.embedder = self.build_embedding_network()
            self.recovery = self.build_recovery_network()
            self.supervisor = self.build_supervisor()
            self.autoencoder = self.construct_autoencoder()

        self.adversarial_supervised, self.adversarial_embedded = self.construct_adversarial()
        self.final_g_model = self.construct_final_g_model()
        self.final_d_model = self.construct_final_d_model()
        self._mse = MeanSquaredError()
        self._bce = BinaryCrossentropy()
        self._scce = SparseCategoricalCrossentropy()
        self.train()
 
    def save_process(self, epoch):
        self.final_g_model.save(PATH + "g_model_fin.h5".format(epoch))
        self.final_d_model.save(PATH + "d_model_fin.h5".format(epoch))

    def construct_autoencoder(self):
        H = self.embedder.output
        X_tilde = self.recovery(H)
        return Model(inputs=self.embedder.input, outputs=X_tilde)

    def construct_adversarial(self):
        E_Hat  = self.g_model.output
        H_hat = self.supervisor(E_Hat)
        Y_fake, Y_fake_cate = self.d_model(H_hat)

        adversarial_supervised = Model(inputs=self.g_model.input, outputs=[Y_fake, Y_fake_cate], name='AdversarialSupervised') 

        Y_fake_e, Y_fake_cate_e = self.d_model(E_Hat)
        adversarial_embedded = Model(inputs=self.g_model.input, outputs=[Y_fake_e, Y_fake_cate_e], name='AdversarialEmbedded')
        return adversarial_supervised, adversarial_embedded      

    def construct_final_g_model(self):
        E_Hat = self.g_model.output
        H_hat = self.supervisor(E_Hat)
        X_hat = self.recovery(H_hat)
        return Model(inputs=self.g_model.input, outputs=X_hat, name='FinalGenerator')    

    def construct_final_d_model(self):
        Y_real = self.d_model(self.embedder.output)
        return Model(inputs=self.embedder.input, outputs=Y_real, name="RealDiscriminator")            

    def build_supervisor(self):
        in_image = Input(shape=[self.data.shape[1], self.hidden_dim])

        emb = GRU(units=128, return_sequences=True,activation='tanh')(in_image)
        emb = GRU(units=64, return_sequences=True, activation='tanh')(emb)
        output = Dense(units=self.hidden_dim, activation='sigmoid')(emb)

        model = Model(in_image, output, name="supervisor")
        model.summary()
        return model

    def build_embedding_network(self):
        data_shape = self.data.shape[1:]
        in_image = Input(shape=data_shape)

        emb = GRU(units=128, return_sequences=True, activation='tanh')(in_image)
        emb = GRU(units=64, return_sequences=True, activation='tanh')(emb)
        output = Dense(units=self.hidden_dim, activation='sigmoid')(emb)

        model = Model(in_image, output, name="embedding")
        model.summary()
        return model

    def build_recovery_network(self):
        in_image = Input(shape=[self.data.shape[1], self.hidden_dim])

        rec = GRU(units=128, return_sequences=True, activation='tanh')(in_image)
        rec = GRU(units=64, return_sequences=True, activation='tanh')(rec)
        output = Dense(units=self.data.shape[-1], activation='sigmoid')(rec)

        model = Model(in_image, output, name="recovery")
        model.summary()
        return model

    def build_discriminator(self):
        # image input
        in_image = Input(shape=[self.data.shape[1], self.hidden_dim])

        fe = GRU(128, return_sequences=True, activation='tanh')(in_image)
        fe = GRU(64, activation='tanh')(fe)

        # determine real/fake
        out1 = Dense(1)(fe)

        # determine class
        out2 = Dense(self.n_classes, activation='softmax')(fe)
        output = [out1, out2]

        model = Model(in_image, output)
        model.summary()
        return model

    def build_generator(self):
        in_label = Input(shape=(1,))
        li = Embedding(self.n_classes, 50)(in_label)
        li = Dense(self.latent_dim)(li)
        li = Reshape((self.latent_dim, 1))(li)

        in_lat = Input(shape=(self.latent_dim,))

        inp = Reshape((self.latent_dim, 1))(in_lat)
        gen = GRU(128, return_sequences=True, activation='tanh')(inp)
        gen = GRU(128, return_sequences=True, activation='tanh')(gen)

        merge = Concatenate()([gen, li])
        gen = Dense(512)(merge)
        gen = Reshape((2048, self.hidden_dim))(gen)

        out_layer = Activation('sigmoid')(gen)

        # define model
        model = Model([in_lat, in_label], out_layer)
        model.summary()
        return model

    def train(self):
        n_epoch = 1000
        learning_rate = 1e-3
        autoencoder_opt = Adam(lr=learning_rate)
        generator_opt = Adam(learning_rate=learning_rate)
        embedder_opt = Adam(learning_rate=learning_rate)
        discriminator_opt = Adam(learning_rate=learning_rate)
        supervisor_opt = Adam(learning_rate=learning_rate)

        def _generate_noise():
            while True:
                yield np.random.uniform(low=0, high=1, size=(self.latent_dim, 1))

        def get_batch_noise():
            return iter(tfdata.Dataset.from_generator(_generate_noise, output_types=float)
                                    .batch(64)
                                    .repeat())

        def get_batch_data(data, n_windows):
            tensor_data = convert_to_tensor(data, dtype=float)
            return iter(tfdata.Dataset.from_tensor_slices(tensor_data)
                                    .shuffle(buffer_size=n_windows)
                                    .batch(64).repeat())
        def get_batch_label(label, n_windows):
            tensor_data = convert_to_tensor(label)
            return iter(tfdata.Dataset.from_tensor_slices(tensor_data)
                                    .shuffle(buffer_size=n_windows)
                                    .batch(64).repeat())

        if self.is_train == False:
            for _ in tqdm(range(n_epoch)):
                X_ = next(get_batch_data(self.data, n_windows=self.data.shape[0]))
                step_e_loss_t0 = self.train_autoencoder(X_, autoencoder_opt)

            self.autoencoder.save(PATH + "autoencoder.h5")
            self.embedder.save(PATH + "emb_layer.h5")
            self.recovery.save(PATH + "rec_layer.h5")

            ## Supervised Network training
            for _ in tqdm(range(n_epoch)):
                X_ = next(get_batch_data(self.data, n_windows=self.data.shape[0]))
                step_g_loss_s = self.train_supervisor(X_, supervisor_opt)
            
            self.supervisor.save(PATH + "supervisor.h5")

        ## Joint training
        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0

        for epoch in tqdm(range(n_epoch)):
            X_ = next(get_batch_data(self.data, n_windows=self.data.shape[0]))
            Z_ = next(get_batch_noise())
            label_ = next(get_batch_label(self.label, n_windows=self.data.shape[0]))

            step_g_loss_u, step_g_loss_s, step_g_loss_v = self.train_generator(X_, label_, Z_, generator_opt)
            step_e_loss_t0 = self.train_embedder(X_, embedder_opt)
            step_d_loss = self.train_discriminator(X_, label_, Z_, discriminator_opt)

            if epoch % 50 == 0:
                g_loss = 0.33 * step_g_loss_u + step_g_loss_s + step_g_loss_v
                print("epoch = {}, step_g_loss = {}, step_e_loss = {}, step_d_loss = {}".format(
                    epoch, g_loss, step_e_loss_t0, step_d_loss))

                self.save_process(epoch)

    @function
    def train_embedder(self, x, opt):
        with GradientTape() as tape:
            # Supervised Loss
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

            # Reconstruction Loss
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss = 10 * sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    @function
    def train_discriminator(self, x, label, z, opt):
        with GradientTape() as tape:
            discriminator_loss = self.discriminator_loss(x, label, z)

        var_list = self.final_d_model.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return discriminator_loss

    @function
    def train_generator(self, x, label, z, opt):
        with GradientTape() as tape:
            y_fake, y_fake_cate = self.adversarial_supervised([z, label])

            generator_loss_unsupervised = self._bce(y_true=ones_like(y_fake),
                                                    y_pred=y_fake)

            generator_loss_unsupervised_cate = self._scce(y_true=label,
                                                    y_pred=y_fake_cate)

            y_fake_e, y_fake_e_cate = self.adversarial_embedded([z, label])
            generator_loss_unsupervised_e = self._bce(y_true=ones_like(y_fake_e),
                                                      y_pred=y_fake_e)

            generator_loss_unsupervised_e_cate = self._scce(y_true=label,
                                                      y_pred=y_fake_e_cate)

            gen_loss_unsu = generator_loss_unsupervised + generator_loss_unsupervised_cate
            gen_loss_unsu_e = generator_loss_unsupervised_e + generator_loss_unsupervised_e_cate

            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

            x_hat = self.final_g_model([z, label])
            generator_moment_loss = self.calc_generator_moments_loss(x, x_hat)

            generator_loss = (0.5 * gen_loss_unsu +
                              0.5 * gen_loss_unsu_e +
                              100 * sqrt(generator_loss_supervised) +
                              100 * generator_moment_loss)

        var_list = self.g_model.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return gen_loss_unsu, generator_loss_supervised, gen_loss_unsu_e

    @function
    def train_autoencoder(self, x, opt):
        with GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss_0 = 10 * sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    @function
    def train_supervisor(self, x, opt):
        with GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

        var_list = self.supervisor.trainable_variables + self.final_g_model.trainable_variables
        gradients = tape.gradient(generator_loss_supervised, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return generator_loss_supervised

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        y_true_mean, y_true_var = nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = nn.moments(x=y_pred, axes=[0])
        g_loss_mean = reduce_mean(abs(y_true_mean - y_pred_mean))
        g_loss_var = reduce_mean(abs(sqrt(y_true_var + 1e-6) - sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var

    def discriminator_loss(self, x, label, z):
        # Loss on false negatives
        y_real, y_real_cate = self.final_d_model(x)
        discriminator_loss_real = self._bce(y_true=ones_like(y_real),
                                            y_pred=y_real)

        discriminator_loss_real_cate = self._scce(y_true=label,
                                            y_pred=y_real_cate)

        # Loss on false positives
        y_fake, y_fake_cate = self.adversarial_supervised([z, label])
        discriminator_loss_fake = self._bce(y_true=zeros_like(y_fake),
                                            y_pred=y_fake)

        discriminator_loss_fake_cate = self._scce(y_true=label,
                                            y_pred=y_fake_cate)

        y_fake_e, y_fake_e_cate = self.adversarial_embedded([z, label])

        discriminator_loss_fake_e = self._bce(y_true=zeros_like(y_fake_e),
                                              y_pred=y_fake_e)

        discriminator_loss_fake_e_cate = self._scce(y_true=label,
                                              y_pred=y_fake_e_cate)

        alpha = 0.4
        total_d_loss_real = alpha * discriminator_loss_real + (1-alpha) * discriminator_loss_real_cate
        total_d_loss_fake = alpha * discriminator_loss_fake + (1-alpha) * discriminator_loss_fake_cate
        total_d_loss_fake_e = alpha * discriminator_loss_fake_e + (1-alpha)*discriminator_loss_fake_e_cate

        return (total_d_loss_real + total_d_loss_fake + total_d_loss_fake_e)

if __name__ == "__main__":
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    scaler = MinMaxScaler()
    scaled_x = []

    for x in X_train:
        x = scaler.fit_transform(x)
        scaled_x.append(x)

    X_train = np.array(scaled_x)
    del scaled_x
    TSGAN(X_train, Y_train, is_train=False)