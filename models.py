import numpy as np 
import tensorflow as tf 

def get_model(dropout=0.):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(784,)))
    if dropout > 0.:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(10))
    return model

class Baseline():
    def __init__(self): 
        self.model = get_model()
        self.reset_metrics()
        self.optimizer = tf.keras.optimizers.SGD(1e-2)
        self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
    def reset_metrics(self):
        self.metrics = {
            'w_norm': [],
            'w_mean': [],
            'w_var': [],
            'w_rank': [],
            'wg_norm': [],
            'wg_mean': [],
            'wg_var': [],
            'loss': []
        }

    def train(self, xb, yb):
        with tf.GradientTape() as g: 
            y_preds = self.model(xb, training=True)
            loss = self.loss_func(yb, y_preds)
        grads = g.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.record_metrics(grads, loss)

    def record_metrics(self, grads, loss):
        W = self.model.trainable_weights[0]
        w_norm = tf.norm(W, 2)
        w_mean, w_var = tf.nn.moments(W, [0, 1])
        w_rank = tf.linalg.matrix_rank(W)
        wg_norm = tf.norm(grads[0], 2)
        wg_mean, wg_var = tf.nn.moments(grads[0], [0, 1])

        self.metrics['w_norm'].append(w_norm)
        self.metrics['w_mean'].append(w_mean)
        self.metrics['w_var'].append(w_var)
        self.metrics['w_rank'].append(w_rank)
        self.metrics['wg_norm'].append(wg_norm)
        self.metrics['wg_mean'].append(wg_mean)
        self.metrics['wg_var'].append(wg_var)
        self.metrics['loss'].append(loss)

    def get_metrics(self):
        mean_metrics = {}
        for k in self.metrics.keys(): 
            mean_metrics[k] = np.mean(self.metrics[k])
        self.reset_metrics()
        return mean_metrics


class LipschitzReg(Baseline):
    def __init__(self, reg_constant):
        super().__init__()
        self.reg_constant = reg_constant

    def train(self, xb, yb):
        with tf.GradientTape() as gg:
            with tf.GradientTape() as g: 
                y_pred = self.model(xb)
                loss = self.loss_func(yb, y_pred)
            grads = g.gradient(loss, self.model.trainable_weights)
            # compute lipschitz term gradients 
            gg_grads = gg.gradient([self.reg_constant * (tf.norm(v, 2) - 1) ** 2 for v in grads], self.model.trainable_weights)

        # optimize off of loss + lipschitz gradients 
        final_grads = [g + gg for g, gg in zip(grads, gg_grads)]
        self.optimizer.apply_gradients(zip(final_grads, self.model.trainable_weights))
        
        self.record_metrics(final_grads, loss)

class DropoutReg(Baseline):
    def __init__(self, dropout_constant): 
        super().__init__()
        self.model = get_model(dropout_constant)


class SpectralReg(Baseline):
    def __init__(self, reg_constant):
        super().__init__()
        self.reg_constant = reg_constant
        self.v = tf.random.normal((10, 1), mean=0., stddev=1.)

    def train(self, xb, yb):
        with tf.GradientTape() as g: 
            y_preds = self.model(xb, training=True)
            loss = self.loss_func(yb, y_preds)
        grads = g.gradient(loss, self.model.trainable_weights)
        
        # apply spectral norm reg. 
        W = self.model.trainable_weights[0]
        W_grad = grads[0]
        u = W @ self.v 
        self.v = tf.transpose(W) @ u 
        sigma = tf.norm(u, 2) / tf.norm(self.v, 2)
        reg_value = sigma * (u @ tf.transpose(self.v))
        W_grad += self.reg_constant * reg_value
        grads[0] = W_grad 

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.record_metrics(grads, loss)

class OrthogonalReg(Baseline):
    def __init__(self, reg_constant):
        super().__init__()
        self.reg_constant = reg_constant
        self.model = self.build_model()

    def build_model(self):
        def orthogonal_reg(W):
            orthog_term = tf.abs(W @ tf.transpose(W) - tf.eye(W.shape.as_list()[0])).sum()
            return self.reg_constant * orthog_term

        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(784,)))
        model.add(tf.keras.layers.Dense(10, kernel_regularizer=orthogonal_reg))
        return model 