
import tensorflow as tf 


def ANN(input_shape, output_shape):
        x_in  = tf.keras.layers.Input(shape = input_shape)
        x = tf.keras.layers.Dense(64, activation='relu')(x_in)
        # x = tf.keras.layers.GRU(32)(x) ## (None, Time ,Features)
        # x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        output = tf.keras.layers.Dense(output_shape, activation = 'linear')(x)
        
        model = tf.keras.models.Model(inputs = x_in,
                                           outputs = output)
        
        return model 
        

        
class ANNasClass():
    def  __init__(self,input_shape, output_shape):
        self.output_shape = output_shape
        self.input_shape = input_shape
        
        x_in  = tf.keras.layers.Input(shape = input_shape)
        x = tf.keras.layers.Dense(64, activation='relu')(x_in)
        # x = tf.keras.layers.GRU(32)(x) ## (None, Time ,Features)
        # x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        output = tf.keras.layers.Dense(output_shape, activation = 'linear')(x)
        
        self.model = tf.keras.models.Model(inputs = x_in,
                                           outputs = output)
        
        ##activate net 
        x = tf.random.normal(shape = input_shape)
        x = tf.expand_dims(x, axis=0)
        self.__call__(x)
        
        self.trainable_variables = []
        self.trainable_variables +=self.model.trainable_variables
        
    def __call__(self,x):
        # print("yes:", x)
        # input("df")
        return self.model(x)
  
class ANNasClassDueling():
    def  __init__(self,input_shape, output_shape):
        self.output_shape = output_shape
        self.input_shape = input_shape
        
        x_in  = tf.keras.layers.Input(shape = input_shape)
        x = tf.keras.layers.Dense(64, activation='relu')(x_in)
        # x = tf.keras.layers.GRU(32)(x) ## (None, Time ,Features)
        # x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        
        val = tf.keras.layers.Dense(1, activation=tf.identity)(x)
        adv = tf.keras.layers.Dense(output_shape, activation = tf.identity)(x)
     
        ## Q(s,a) = V(s) + A(s,a)  - 1/|A| * sum(A(s,a_i))

        mean_adv = tf.reduce_mean(adv, -1, keepdims=True)
        
        q_s_a = val + adv - mean_adv
        
        self.model = tf.keras.models.Model(inputs = x_in,
                                           outputs = q_s_a)
        
        ##activate net 
        x = tf.random.normal(shape = input_shape)
        x = tf.expand_dims(x, axis=0)
        self.__call__(x)
        
        self.trainable_variables = []
        self.trainable_variables +=self.model.trainable_variables
        
    def __call__(self,x):
        # print("yes:", x)
        # input("df")
        return self.model(x)
              
