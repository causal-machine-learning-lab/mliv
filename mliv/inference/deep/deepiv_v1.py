# from __future__ import absolute_import, division, print_function, unicode_literals

import keras
import types
import random
import numpy as np
import tensorflow as tf
from mliv.utils import set_seed

from keras import backend as K
from keras.layers import Input, Dense, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D, Lambda
from keras.models import Model, Sequential
from keras.layers.merge import Concatenate
from keras.backend import clear_session
from keras.engine.topology import InputLayer
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.layers.core import Reshape

if K.backend() == "theano":
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    import theano.tensor as tensor
    _FLOATX = theano.config.floatX
    Lop = tensor.Lop
elif K.backend() == "tensorflow":
    def Lop(output, wrt, eval_points):
        grads = tf.gradients(output, wrt, grad_ys=eval_points)
        return grads

example = '''
from mliv.inference import DeepIV

model = DeepIV()
model.fit(data)
ITE = model.predict(data.train)
ATE,_ = model.ATE(data.train)
'''

############ from DeepIV_True.custom_gradients import replace_gradients_mse

def get_gradients(self, loss, params):
    '''
    Replacement for the default keras get_gradients() function.
    Modification: checks if the object has the attribute grads and 
    returns that rather than calculating the gradients using automatic
    differentiation. 
    '''
    if hasattr(self, 'grads'):
        grads = self.grads
    else:
        grads = K.gradients(loss, params)
    if hasattr(self, 'clipnorm') and self.clipnorm > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
    if hasattr(self, 'clipvalue') and self.clipvalue > 0:
        grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
    return grads

def replace_gradients_mse(model, opt, batch_size, n_samples = 1):
    '''
    Replace the gradients of a Keras model with mean square error loss.
    '''
    # targets has been repeated twice so the below creates two identical columns
    # of the target values - we'll only use the first column.
    targets = K.reshape(model.targets[0], (batch_size, n_samples * 2))
    output =  K.mean(K.reshape(model.outputs[0], (batch_size, n_samples, 2)), axis=1)
    # compute d Loss / d output
    dL_dOutput = (output[:,0] - targets[:,0]) * (2.) / batch_size
    # compute (d Loss / d output) (d output / d theta) for each theta
    trainable_weights = model.trainable_weights
    grads = Lop(output[:,1], wrt=trainable_weights, eval_points=dL_dOutput) 
    # compute regularizer gradients

    # add loss with respect to regularizers
    reg_loss = model.total_loss * 0.
    for r in model.losses:
         reg_loss += r
    reg_grads = K.gradients(reg_loss, trainable_weights)
    grads = [g+r for g,r in zip(grads, reg_grads)]
    
    opt = keras.optimizers.get(opt)
    # Patch keras gradient calculation to allow for user defined gradients
    opt.get_gradients = types.MethodType( get_gradients, opt )
    opt.grads = grads
    model.optimizer = opt
    return model

######### import DeepIV_True.densities as densities

def split(start, stop):
    return Lambda(lambda x: x[:, start:stop], output_shape=(None, stop-start))

def split_mixture_of_gaussians(x, n_components):
    pi = split(0, n_components)(x)
    mu = split(n_components, 2*n_components)(x)
    log_sig = split(2*n_components, 3*n_components)(x)
    return pi, mu, log_sig

def log_norm_pdf(x, mu, log_sig):
    z = (x - mu) / (K.exp(K.clip(log_sig, -40, 40))) #TODO: get rid of this clipping
    return -(0.5)*K.log(2*np.pi) - log_sig - 0.5*((z)**2)

def mix_gaussian_loss(x, mu, log_sig, w):
    '''
    Combine the mixture of gaussian distribution and the loss into a single function
    so that we can do the log sum exp trick for numerical stability...
    '''
    if K.backend() == "tensorflow":
        x.set_shape([None, 1])
    gauss = log_norm_pdf(K.repeat_elements(x=x, rep=mu.shape[1], axis=1), mu, log_sig)
    # TODO: get rid of clipping.
    gauss = K.clip(gauss, -40, 40)
    max_gauss = K.maximum((0.), K.max(gauss))
    # log sum exp trick...
    gauss = gauss - max_gauss
    out = K.sum(w * K.exp(gauss), axis=1)
    loss = K.mean(-K.log(out) + max_gauss)
    return loss

def mixture_of_gaussian_output(x, n_components):
    mu = keras.layers.Dense(n_components, activation='linear')(x)
    log_sig = keras.layers.Dense(n_components, activation='linear')(x)
    pi = keras.layers.Dense(n_components, activation='softmax')(x)
    return Concatenate(axis=1)([pi, mu, log_sig])

def mixture_of_gaussian_loss(y_true, y_pred, n_components):
    pi, mu, log_sig = split_mixture_of_gaussians(y_pred, n_components)
    return mix_gaussian_loss(y_true, mu, log_sig, pi)

######### import DeepIV_True.samplers as samplers

def random_laplace(shape, mu=0., b=1.):
    '''
    Draw random samples from a Laplace distriubtion.

    See: https://en.wikipedia.org/wiki/Laplace_distribution#Generating_random_variables_according_to_the_Laplace_distribution
    '''
    U = K.random_uniform(shape, -0.5, 0.5)
    return mu - b * K.sign(U) * K.log(1 - 2 * K.abs(U))

def random_normal(shape, mean=0.0, std=1.0):
    return K.random_normal(shape, mean, std)

def random_multinomial(logits, seed=None):
    '''
    Theano function for sampling from a multinomal with probability given by `logits`
    '''
    if K.backend() == "theano":
        if seed is None:
            seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
        return rng.multinomial(n=1, pvals=logits, ndim=None, dtype=_FLOATX)
    elif K.backend() == "tensorflow":
        return tf.one_hot(tf.squeeze(tf.multinomial(K.log(logits), num_samples=1)),
                          int(logits.shape[1]))

def random_gmm(pi, mu, sig):
    '''
    Sample from a gaussian mixture model. Returns one sample for each row in
    the pi, mu and sig matrices... this is potentially wasteful (because you have to repeat
    the matrices n times if you want to get n samples), but makes it easy to implment
    code where the parameters vary as they are conditioned on different datapoints.
    '''
    normals = random_normal(K.shape(mu), mu, sig)
    k = random_multinomial(pi)
    return K.sum(normals * k, axis=1, keepdims=True)


######### from DeepIV_True.models import Treatment, Response

class Treatment(Model):
    '''
    Adds sampling functionality to a Keras model and extends the losses to support
    mixture of gaussian losses.

    # Argument
    '''

    def _get_sampler_by_string(self, loss):
        output = self.outputs[0]
        inputs = self.inputs

        if loss in ["MSE", "mse", "mean_squared_error"]:
            output += random_normal(K.shape(output), mean=0.0, std=1.0)
            draw_sample = K.function(inputs + [K.learning_phase()], [output])

            def sample_gaussian(inputs, use_dropout=False):
                '''
                Helper to draw samples from a gaussian distribution
                '''
                return draw_sample(inputs + [int(use_dropout)])[0]

            return sample_gaussian

        elif loss == "binary_crossentropy":
            output = K.random_binomial(K.shape(output), p=output)
            draw_sample = K.function(inputs + [K.learning_phase()], [output])

            def sample_binomial(inputs, use_dropout=False):
                '''
                Helper to draw samples from a binomial distribution
                '''
                return draw_sample(inputs + [int(use_dropout)])[0]

            return sample_binomial

        elif loss in ["mean_absolute_error", "mae", "MAE"]:
            output += random_laplace(K.shape(output), mu=0.0, b=1.0)
            draw_sample = K.function(inputs + [K.learning_phase()], [output])
            def sample_laplace(inputs, use_dropout=False):
                '''
                Helper to draw samples from a Laplacian distribution
                '''
                return draw_sample(inputs + [int(use_dropout)])[0]

            return sample_laplace

        elif loss == "mixture_of_gaussians":
            pi, mu, log_sig = split_mixture_of_gaussians(output, self.n_components)
            samples = random_gmm(pi, mu, K.exp(log_sig))
            draw_sample = K.function(inputs + [K.learning_phase()], [samples])
            return lambda inputs, use_dropout: draw_sample(inputs + [int(use_dropout)])[0]

        else:
            raise NotImplementedError("Unrecognised loss: %s. Cannot build a generic sampler" % loss)

    def _prepare_sampler(self, loss):
        '''
        Build sampler
        '''
        if isinstance(loss, str):
            self._sampler = self._get_sampler_by_string(loss)
        else:
            warnings.warn("You're using a custom loss function. Make sure you implement\
                           the model's sample() fuction yourself.")

    def compile(self, optimizer, loss, metrics=None, loss_weights=None,
                sample_weight_mode=None, n_components=None, **kwargs):
        '''
        Overrides the existing keras compile function to add a sampler building
        step to the model compilation phase. Once compiled, one can draw samples
        from the network using the sample() function and adds support for mixture
        of gaussian loss.

        '''
        if loss == "mixture_of_gaussians":
            if n_components is None:
                raise Exception("When using mixture of gaussian loss you must\
                                 supply n_components argument")
            self.n_components = n_components
            self._prepare_sampler(loss)
            loss = lambda y_true, y_pred: mixture_of_gaussian_loss(y_true,y_pred,n_components)
        else:
            self._prepare_sampler(loss)

        super(Treatment, self).compile(optimizer, loss, metrics=metrics, loss_weights=loss_weights,
                                       sample_weight_mode=sample_weight_mode, **kwargs)

    def sample(self, inputs, n_samples=1, use_dropout=False):
        '''
        Draw samples from the keras model.
        '''
        if hasattr(self, "_sampler"):
            if not isinstance(inputs, list):
                inputs = [inputs]
            inputs = [i.repeat(n_samples, axis=0) for i in inputs]
            return self._sampler(inputs, use_dropout)
        else:
            raise Exception("Compile model with loss before sampling")

class Response(Model):
    '''
    Extends the Keras Model class to support sampling from the Treatment
    model during training.

    Overwrites the existing fit_generator function.

    # Arguments
    In addition to the standard model arguments, a Response object takes
    a Treatment object as input so that it can sample from the fitted treatment
    distriubtion during training.
    '''
    def __init__(self, treatment, **kwargs):
        if isinstance(treatment, Treatment):
            self.treatment = treatment
        else:
            raise TypeError("Expected a treatment model of type Treatment. \
                             Got a model of type %s. Remember to train your\
                             treatment model first." % type(treatment))
        super(Response, self).__init__(**kwargs)

    def compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None,
                unbiased_gradient=False,n_samples=1, batch_size=None):
        super(Response, self).compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights,
                                      sample_weight_mode=sample_weight_mode)
        self.unbiased_gradient = unbiased_gradient
        if unbiased_gradient:
            if loss in ["MSE", "mse", "mean_squared_error"]:
                if batch_size is None:
                    raise ValueError("Must supply a batch_size argument if using unbiased gradients. Currently batch_size is None.")
                replace_gradients_mse(self, optimizer, batch_size=batch_size, n_samples=n_samples)
            else:
                warnings.warn("Unbiased gradient only implemented for mean square error loss. It is unnecessary for\
                              logistic losses and currently not implemented for absolute error losses.")
            

    def fit(self, x=None, y=None, batch_size=512, epochs=1, verbose=1, callbacks=None,
            validation_data=None, class_weight=None, initial_epoch=0, samples_per_batch=None,
            seed=None, observed_treatments=None):
        '''
        Trains the model by sampling from the fitted treament distribution.

        # Arguments
            x: list of numpy arrays. The first element should *always* be the instrument variables.
            y: (numpy array). Target response variables.
            The remainder of the arguments correspond to the Keras definitions.
        '''
        batch_size = np.minimum(y.shape[0], batch_size)
        if seed is None:
            seed = np.random.randint(0, 1e6)
        if samples_per_batch is None:
            if self.unbiased_gradient:
                samples_per_batch = 2
            else:
                samples_per_batch = 1

        if observed_treatments is None:
            generator = SampledSequence(x[1:], x[0], y, batch_size, self.treatment.sample, samples_per_batch)
        else:
            generator = OnesidedUnbaised(x[1:], x[0], y, observed_treatments, batch_size,
                                         self.treatment.sample, samples_per_batch)
        
        steps_per_epoch = y.shape[0]  // batch_size
        super(Response, self).fit_generator(generator=generator,
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=epochs, verbose=verbose,
                                            callbacks=callbacks, validation_data=validation_data,
                                            class_weight=class_weight, initial_epoch=initial_epoch)

    def fit_generator(self, **kwargs):
        '''
        We use override fit_generator to support sampling from the treatment model during training.

        If you need this functionality, you'll need to build a generator that samples from the
        treatment and performs whatever transformations you're performing. Please submit a pull
        request if you implement this.
        '''
        raise NotImplementedError("We use override fit_generator to support sampling from the\
                                   treatment model during training.")

    def expected_representation(self, x, z, n_samples=100, batch_size=None, seed=None):
        inputs = [z, x]
        if not hasattr(self, "_E_representation"):
            if batch_size is None:
                batch_size = inputs[0].shape[0]
                steps = 1
            else:
                steps = inputs[0].shape[0] // batch_size

            intermediate_layer_model = Model(inputs=self.inputs,
                                             outputs=self.layers[-2].output)
            
            def pred(inputs, n_samples=100, seed=None):
                features = inputs[1]

                samples = self.treatment.sample(inputs, n_samples)
                batch_features = [features.repeat(n_samples, axis=0)] + [samples]
                representation = intermediate_layer_model.predict(batch_features)
                return representation.reshape((inputs[0].shape[0], n_samples, -1)).mean(axis=1)
            self._E_representation = pred
            return self._E_representation(inputs, n_samples, seed)
        else:
            return self._E_representation(inputs, n_samples, seed)

    def conditional_representation(self, x, p):
        inputs = [x, p]
        if not hasattr(self, "_c_representation"):          
            intermediate_layer_model = Model(inputs=self.inputs,
                                             outputs=self.layers[-2].output)

            self._c_representation = intermediate_layer_model.predict
            return self._c_representation(inputs)
        else:
            return self._c_representation(inputs)

    def dropout_predict(self, x, z, n_samples=100):
        if isinstance(x, list):
            inputs = [z] + x
        else:
            inputs = [z, x]
        if not hasattr(self, "_dropout_predict"):
            
            predict_with_dropout = K.function(self.inputs + [K.learning_phase()],
                                              [self.layers[-1].output])

            def pred(inputs, n_samples = 100):
                # draw samples from the treatment network with dropout turned on
                samples = self.treatment.sample(inputs, n_samples, use_dropout=True)
                # prepare inputs for the response network
                rep_inputs = [i.repeat(n_samples, axis=0) for i in inputs[1:]] + [samples]
                # return outputs from the response network with dropout turned on (learning_phase=0)
                return predict_with_dropout(rep_inputs + [1])[0]
            self._dropout_predict = pred
            return self._dropout_predict(inputs, n_samples)
        else:
            return self._dropout_predict(inputs, n_samples)

    def credible_interval(self, x, z, n_samples=100, p=0.95):
        '''
        Return a credible interval of size p using dropout variational inference.
        '''
        if isinstance(x, list):
            n = x[0].shape[0]
        else:
            n = x.shape[0]
        alpha = (1-p) / 2.
        samples = self.dropout_predict(x, z, n_samples).reshape((n, n_samples, -1))
        upper = np.percentile(samples.copy(), 100*(p+alpha), axis=1)
        lower = np.percentile(samples.copy(), 100*(alpha), axis=1)
        return lower, upper

    def _add_constant(self, X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    
    def predict_confidence(self, x, p):
        if hasattr(self, "_predict_confidence"):
            return self._predict_confidence(x, p)
        else:
            raise Exception("Call fit_confidence_interval before running predict_confidence")

    
    def fit_confidence_interval(self, x_lo, z_lo, p_lo, y_lo, n_samples=100, alpha=0.):
        eta_bar = self.expected_representation(x=x_lo, z=z_lo, n_samples=n_samples)
        pca = PCA(1-1e-16, svd_solver="full", whiten=True)
        pca.fit(eta_bar)

        eta_bar = pca.transform(eta_bar)
        eta_lo_prime = pca.transform(self.conditional_representation(x_lo, p_lo))
        eta_lo = self._add_constant(eta_lo_prime)

        ols1 = linear_model.Ridge(alpha=alpha, fit_intercept=True)
        ols1.fit(eta_bar, eta_lo_prime)
        hhat = ols1.predict(eta_bar)
        ols2 = linear_model.Ridge(alpha=alpha, fit_intercept=False)
        ols2.fit(self._add_constant(hhat), y_lo)

        yhat = ols2.predict(eta_lo)
        hhi = np.linalg.inv(np.dot(eta_lo.T, eta_lo))

        heh = np.dot(eta_lo.T, np.square(y_lo - yhat) * eta_lo)
        V = np.dot(np.dot(hhi, heh), hhi)

        def pred(xx, pp):
            H = self._add_constant(pca.transform(self.conditional_representation(xx,pp)))
            sdhb = np.sqrt(np.diag(np.dot(np.dot(H, V), H.T)))
            hb = ols2.predict(H).flatten()
            return hb, sdhb
        
        self._predict_confidence = pred

class SampledSequence(keras.utils.Sequence):
    def __init__(self, features, instruments, outputs, batch_size, sampler, n_samples=1, seed=None):
        self.rng = np.random.RandomState(seed)
        if not isinstance(features, list):
            features = [features.copy()]
        else:
            features = [f.copy() for f in features]
        self.features = features
        self.instruments = instruments.copy()
        self.outputs = outputs.copy()
        if batch_size < self.instruments.shape[0]:
            self.batch_size = batch_size
        else:
            self.batch_size = self.instruments.shape[0]
        self.sampler = sampler
        self.n_samples = n_samples
        self.current_index = 0
        self.shuffle()

    def __len__(self):
        if isinstance(self.outputs, list):
            return self.outputs[0].shape[0] // self.batch_size
        else:
            return self.outputs.shape[0] // self.batch_size

    def shuffle(self):
        idx = self.rng.permutation(np.arange(self.instruments.shape[0]))
        self.instruments = self.instruments[idx,:]
        self.outputs = self.outputs[idx,:]
        self.features = [f[idx,:] for f in self.features]
    
    def __getitem__(self,idx):
        instruments = [self.instruments[idx*self.batch_size:(idx+1)*self.batch_size, :]]
        features = [inp[idx*self.batch_size:(idx+1)*self.batch_size, :] for inp in self.features]
        sampler_input = instruments + features
        samples = self.sampler(sampler_input, self.n_samples)
        batch_features = [f[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0) for f in self.features] + [samples]
        batch_y = self.outputs[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0)
        if idx == (len(self) - 1):
            self.shuffle()
        return batch_features, batch_y

class OnesidedUnbaised(SampledSequence):
    def __init__(self, features, instruments, outputs, treatments, batch_size, sampler, n_samples=1, seed=None):
        self.rng = np.random.RandomState(seed)
        if not isinstance(features, list):
            features = [features.copy()]
        else:
            features = [f.copy() for f in features]
        self.features = features
        self.instruments = instruments.copy()
        self.outputs = outputs.copy()
        self.treatments = treatments.copy()
        self.batch_size = batch_size
        self.sampler = sampler
        self.n_samples = n_samples
        self.current_index = 0
        self.shuffle()

    def shuffle(self):
        idx = self.rng.permutation(np.arange(self.instruments.shape[0]))
        self.instruments = self.instruments[idx,:]
        self.outputs = self.outputs[idx,:]
        self.features = [f[idx,:] for f in self.features]
        self.treatments = self.treatments[idx,:]

    def __getitem__(self, idx):
        instruments = [self.instruments[idx*self.batch_size:(idx+1)*self.batch_size, :]]
        features = [inp[idx*self.batch_size:(idx+1)*self.batch_size, :] for inp in self.features]
        observed_treatments = self.treatments[idx*self.batch_size:(idx+1)*self.batch_size, :]
        sampler_input = instruments + features
        samples = self.sampler(sampler_input, self.n_samples // 2)
        samples = np.concatenate([observed_treatments, samples], axis=0)
        batch_features = [f[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0) for f in self.features] + [samples]
        batch_y = self.outputs[idx*self.batch_size:(idx+1)*self.batch_size].repeat(self.n_samples, axis=0)
        if idx == (len(self) - 1):
            self.shuffle()
        return batch_features, batch_y

######### import DeepIV_True.architectures as architectures
#new#
def binary_crossentropy_output(x):
    pi = keras.layers.Dense(1, activation='softmax')(x)
    return pi

def mixture_of_gaussian_output(x, n_components):
    mu = keras.layers.Dense(n_components, activation='linear')(x)
    log_sig = keras.layers.Dense(n_components, activation='linear')(x)
    pi = keras.layers.Dense(n_components, activation='softmax')(x)
    return Concatenate(axis=1)([pi, mu, log_sig])

def feed_forward_net(input, output, hidden_layers=[64, 64], activations='relu',
                     dropout_rate=0., l2=0., constrain_norm=False):
    '''
    Helper function for building a Keras feed forward network.

    input:  Keras Input object appropriate for the data. e.g. input=Input(shape=(20,))
    output: Function representing final layer for the network that maps from the last
            hidden layer to output.
            e.g. if output = Dense(10, activation='softmax') if we're doing 10 class
            classification or output = Dense(1, activation='linear') if we're doing
            regression.
    '''
    state = input
    if isinstance(activations, str):
        activations = [activations] * len(hidden_layers)
    
    for h, a in zip(hidden_layers, activations):
        if l2 > 0.:
            w_reg = keras.regularizers.l2(l2)
        else:
            w_reg = None
        const = maxnorm(2) if constrain_norm else  None
        state = Dense(h, activation=a, kernel_regularizer=w_reg, kernel_constraint=const)(state)
        if dropout_rate > 0.:
            state = Dropout(dropout_rate)(state)
    return output(state)

class DeepIV(object):
    def __init__(self) -> None:
        self.config = {
                    'methodName': 'DeepIV',
                    'dropout': 0.5,
                    'epochs': 10,
                    'batch_size': 100,
                    'n_components': 5,
                    'layers': [128, 64, 32], 
                    'activation': 'relu',
                    'samples_per_batch': 2,
                    't_loss': 'mixture_of_gaussians',
                    'y_loss': 'mse',
                    'seed': 2022,   
                    }

    def set_Configuration(self, config):
        self.config = config

    def fit(self, data, exp=-1, config=None):
        if config is None:
            config = self.config

        set_seed(config['seed'])
        clear_session()
        tf.reset_default_graph()
        random.seed(config['seed'])
        tf.compat.v1.set_random_seed(config['seed'])
        np.random.seed(config['seed'])
        data.numpy()

        config['num'] = data.train.length

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth=True
        sess = tf.Session(config=tfconfig)
        K.set_session(sess)

        dropout_rate = min(1000./(1000. + config['num']), config['dropout'])
        epochs = min(int(1000000./float(config['num'])), config['epochs'])

        instruments = Input(shape=(data.train.z.shape[1],), name="instruments")
        features = Input(shape=(data.train.x.shape[1],), name="features")
        treatment_input = Concatenate(axis=1)([instruments, features])

        est_treat = feed_forward_net(treatment_input, lambda x: mixture_of_gaussian_output(x, config['n_components']),
                                                hidden_layers=config['layers'],
                                                dropout_rate=dropout_rate, l2=0.0001,
                                                activations=config['activation'])


        treatment_model = Treatment(inputs=[instruments, features], outputs=est_treat)
        treatment_model.compile('adam', loss=config['t_loss'], n_components=config['n_components'])

        treatment_model.fit([data.train.z, data.train.x], data.train.t, epochs=epochs, batch_size=config['batch_size'])

        treatment = Input(shape=(data.train.t.shape[1],), name="treatment")
        response_input = Concatenate(axis=1)([features, treatment])

        est_response = feed_forward_net(response_input, Dense(1),
                                                        activations=config['activation'],
                                                        hidden_layers=config['layers'],
                                                        l2=0.001,
                                                        dropout_rate=dropout_rate)

        response_model = Response(treatment=treatment_model,
                                    inputs=[features, treatment],
                                    outputs=est_response)
        response_model.compile('adam', loss=config['y_loss'])

        print('Run {}-th experiment for {}. '.format(exp, config['methodName']))

        response_model.fit([data.train.z, data.train.x], data.train.y, epochs=epochs, verbose=1,
                            batch_size=config['batch_size'], samples_per_batch=config['samples_per_batch'])

        def estimation(data):
            return response_model.predict([data.x, data.t-data.t]), response_model.predict([data.x, data.t])

        print('End. ' + '-'*20)

        self.estimation = estimation
        self.response_model = response_model

    def predict(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        return self.response_model.predict([x, t])

    def ITE(self, data=None, t=None, x=None):
        if data is None:
            data = self.data.test

        if x is None:
            x = data.x

        if t is None:
            t = data.t

        ITE_0 = self.response_model.predict([x, t-t])
        ITE_1 = self.response_model.predict([x, t-t+1])
        ITE_t = self.response_model.predict([x, t])

        return ITE_0,ITE_1,ITE_t

    def ATE(self, data=None, t=None, x=None):
        ITE_0,ITE_1,ITE_t = self.ITE(data,t,x)

        return np.mean(ITE_1-ITE_0), np.mean(ITE_t-ITE_0)

