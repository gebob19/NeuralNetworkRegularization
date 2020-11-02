#%%
import numpy as np 
import tensorflow as tf 

from tqdm.notebook import tqdm 

from models import * 
from writers import NeptuneWriter

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.enable_eager_execution()
print(tf.__version__, tf.executing_eagerly())

# %%
def get_train_test(batch_size=32):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, 784)).astype(np.float32)
    x_test = np.reshape(x_test, (-1, 784)).astype(np.float32)
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    # Reserve 10,000 samples for validation.
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    y_train = y_train[:-10000]
    x_train = x_train[:-10000]

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset, (x_test, y_test)

def evaluate_on_test(trainer):
    # score on test-set 
    acc_metric = tf.keras.metrics.CategoricalAccuracy()

    y_preds = trainer.model.predict(x_test)
    test_loss = trainer.loss_func(y_test, y_preds).numpy()

    acc_metric.reset_states()
    acc_metric.update_state(y_test, y_preds)
    test_acc = acc_metric.result().numpy()

    return test_loss, test_acc

def evalute_on_val(trainer):
    acc_metric = tf.keras.metrics.CategoricalAccuracy()
    losses = []
    for xb, yb in val_dataset:
        y_preds = trainer.model(xb, training=False)
        loss = trainer.loss_func(yb, y_preds).numpy()
        # Update val metrics
        acc_metric.update_state(yb, y_preds)
        losses.append(loss)
    val_acc = acc_metric.result()
    return np.mean(losses), val_acc

#%%
def train(trainer, writer, epochs):
    step = 0
    for e in range(epochs):
        for xb, yb in tqdm(train_dataset): 
            trainer.train(xb, yb)
            
            step += 1
            metrics = {}

            if step % 100 == 0: 
                metrics = trainer.get_metrics()

            # evaluation
            if step % 200 == 0: 
                val_loss, val_acc = evalute_on_val(trainer)
                metrics['val_acc'] = val_acc
                metrics['val_loss'] = val_loss

            if metrics:     
                writer.write(metrics, step)

    test_loss, test_acc = evaluate_on_test(trainer)
    writer.write({'test_acc': test_acc, 'test_loss': test_loss}, 0)

    writer.fin()


# %%
train_dataset, val_dataset, (x_test, y_test) = get_train_test()

trainer = Baseline()
# trainer = LipschitzReg(reg_constant)
# trainer = DropoutReg(0.1)
# trainer = SpectralReg(reg_constant)
# trainer = OrthogonalReg(reg_constant)

#%%
writer = NeptuneWriter('gebob19/672')
train(trainer, writer, 1)


# %%