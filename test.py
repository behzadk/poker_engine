import tensorflow as tf
import tensorflow.keras.backend as K


@tf.function 
def tf_loss_func(): 
  return K.variable((X+1.0)**2)


# Function that calculates the loss
@tf.function
def update_loss(X, loss):
    loss.assign((X+1.0)**2)

def test():
    # Initialise a value of loss
    loss = tf.Variable(0.0)
    print(K.eval(loss))

    # Initialise variable
    X = tf.Variable(2.0)

    # Update loss
    update_loss(X, loss)
    print(K.eval(loss))

    # Change network by adding 1.0
    X.assign_add(1.0)

    # Update loss
    update_loss(X, loss)
    print(K.eval(loss))


if __name__ == "__main__":
    test()