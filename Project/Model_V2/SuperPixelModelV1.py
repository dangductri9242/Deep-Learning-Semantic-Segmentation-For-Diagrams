import tensorflow as tf
from SuperpixelUtilsV3 import SPLocalToGlobal
from SuperpixelUtilsV3 import SPMasks
from SuperpixelUtilsV3 import SPFeaturesMeanVectorized
from SuperpixelUtilsV3 import SPClassAssignmentsVectorized
from SuperpixelUtilsV3 import ConvolutionActivation

def build_basic_spixelnet_normalized():

    # Layer:       Input
    # Input size:  (None, None, 3) images
    # Output size: (None, None, 3) tensor
    inputs = tf.keras.Input(
        shape = (None, None, 3),
        name = 'Inputs')
    
# self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3) #----------------------------------------------------------------------------------------------------------------------------------------------------------
    conv0a_conv = tf.keras.layers.Conv2D(
                filters = 16,
                kernel_size = (3,3),
                padding = 'same',
                activation = tf.nn.relu,
                name = 'conv0a_conv',
                use_bias = False)(inputs)

    # Layer:       BatchNorm1
    conv0a_batchnorm1 = tf.keras.layers.BatchNormalization()(conv0a_conv)

    # Layer:       ReLU
    conv0a_relu = tf.keras.layers.LeakyReLU(0.1)(conv0a_batchnorm1)

# self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3) #----------------------------------------------------------------------------------------------------------------------------------------------------------
    conv0b_conv = tf.keras.layers.Conv2D(
                filters = 16,
                kernel_size = (3,3),
                padding = 'same',
                activation = tf.nn.relu,
                name = 'conv0b_conv',
                use_bias = False)(conv0a_relu)
    
    # Layer:       BatchNorm1
    conv0b_batchnorm1 = tf.keras.layers.BatchNormalization()(conv0b_conv)
    
    # Layer:       ReLU
    conv0b_relu = tf.keras.layers.LeakyReLU(0.1)(conv0b_batchnorm1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
    
# self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
    conv1a_conv = tf.keras.layers.Conv2D(
                filters = 32,
                kernel_size = (3,3),
                padding = 'same',
                strides = 2,
                activation = tf.nn.relu,
                name = 'conv1a_conv',
                use_bias = False)(conv0b_relu)
    
    # Layer:       BatchNorm1
    conv1a_batchnorm1 = tf.keras.layers.BatchNormalization()(conv1a_conv)

    # Layer:       ReLU
    conv1a_relu = tf.keras.layers.LeakyReLU(0.1)(conv1a_batchnorm1)

# self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
    conv1b_conv = tf.keras.layers.Conv2D(
                filters = 32,   
                kernel_size = (3,3),
                padding = 'same',
                activation = tf.nn.relu,
                name = 'conv1b_conv',
                use_bias = False)(conv1a_relu)
    
    # Layer:       BatchNorm1
    conv1b_batchnorm1 = tf.keras.layers.BatchNormalization()(conv1b_conv)

    # Layer:       ReLU
    conv1b_relu = tf.keras.layers.LeakyReLU(0.1)(conv1b_batchnorm1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2) # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    
    conv2a_conv = tf.keras.layers.Conv2D(
                filters = 64,
                kernel_size = (3,3),
                padding = 'same',
                strides = 2,
                activation = tf.nn.relu,
                name = 'conv2a_conv',
                use_bias = False)(conv1b_relu)
    
    # Layer:       BatchNorm1
    conv2a_batchnorm1 = tf.keras.layers.BatchNormalization()(conv2a_conv)

    # Layer:       ReLU
    conv2a_relu = tf.keras.layers.LeakyReLU(0.1)(conv2a_batchnorm1)

# self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3) # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    conv2b_conv = tf.keras.layers.Conv2D(
                filters = 64,
                kernel_size = (3,3),
                padding = 'same',
                activation = tf.nn.relu,
                name = 'conv2b_conv',
                use_bias = False)(conv2a_relu)
    
    # Layer:       BatchNorm1
    conv2b_batchnorm1 = tf.keras.layers.BatchNormalization()(conv2b_conv)

    # Layer:       ReLU
    conv2b_relu = tf.keras.layers.LeakyReLU(0.1)(conv2b_batchnorm1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2# ----------------------------------------------------------------------------------------------------------------------------------------------------------)
#     conv3a_conv = tf.keras.layers.Conv2D(
#                 filters = 128,
#                 kernel_size = (3,3),
#                 padding = 'same',
#                 strides = 2,
#                 activation = tf.nn.relu,
#                 name = 'conv3a_conv',
#                 use_bias = False)(conv2b_relu)
    
#     # Layer:       BatchNorm1
#     conv3a_batchnorm1 = tf.keras.layers.BatchNormalization()(conv3a_conv)

#     # Layer:       ReLU
#     conv3a_relu = tf.keras.layers.LeakyReLU(0.1)(conv3a_batchnorm1)

# # self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#     conv3b_conv = tf.keras.layers.Conv2D(
#                 filters = 128,
#                 kernel_size = (3,3),
#                 padding = 'same',
#                 activation = tf.nn.relu,
#                 name = 'conv3b_conv',
#                 use_bias = False)(conv3a_relu)
    
#     # Layer:       BatchNorm1
#     conv3b_batchnorm1 = tf.keras.layers.BatchNormalization()(conv3b_conv)

#     # Layer:       ReLU
#     conv3b_relu = tf.keras.layers.LeakyReLU(0.1)(conv3b_batchnorm1)

# # ----------------------------------------------------------------------------------------------------------------------------------------------------------
# # self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride= 2) # ----------------------------------------------------------------------------------------------------------------------------------------------------------2)
#     conv4a_conv = tf.keras.layers.Conv2D(
#                 filters = 256,
#                 kernel_size = (3,3),
#                 padding = 'same',
#                 strides = 2,
#                 activation = tf.nn.relu,
#                 name = 'conv4a_conv',
#                 use_bias = False)(conv3b_relu)
    
#     # Layer:       BatchNorm1
#     conv4a_batchnorm1 = tf.keras.layers.BatchNormalization()(conv4a_conv)

#     # Layer:       ReLU
#     conv4a_relu = tf.keras.layers.LeakyReLU(0.1)(conv4a_batchnorm1)

# # self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#     conv4b_conv = tf.keras.layers.Conv2D(
#                 filters = 256,
#                 kernel_size = (3,3),
#                 padding = 'same',
#                 activation = tf.nn.relu,
#                 name = 'conv4b_conv',
#                 use_bias = False)(conv4a_relu)
    
#     # Layer:       BatchNorm1
#     conv4b_batchnorm1 = tf.keras.layers.BatchNormalization()(conv4b_conv)

#     # Layer:       ReLU
#     conv4b_relu = tf.keras.layers.LeakyReLU(0.1)(conv4b_batchnorm1)

# # ----------------------------------------------------------------------------------------------------------------------------------------------------------
# # self.deconv3 = deconv(256, 128)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#     deconv3_deconv = tf.keras.layers.Conv2DTranspose(
#                 filters = 128,
#                 kernel_size = (4,4),
#                 padding = 'same',
#                 activation = tf.nn.relu,
#                 name = 'deconv3_deconv',
#                 strides = (2,2),
#                 use_bias = True)(conv4b_relu)

# # self.conv3_1 = conv(self.batchNorm, 256, 128)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#     concat = tf.keras.layers.Concatenate(axis=3)([conv3b_relu, deconv3_deconv])

#     conv3_1_conv = tf.keras.layers.Conv2D(
#                 filters = 128,
#                 kernel_size = (3,3),
#                 padding = 'same',
#                 activation = tf.nn.relu,
#                 name = 'conv3_1_conv',
#                 use_bias = False)(concat)
    
#     # Layer:       BatchNorm1
#     conv3_1_batchnorm1 = tf.keras.layers.BatchNormalization()(conv3_1_conv)

#     # Layer:       ReLU
#     conv3_1_relu = tf.keras.layers.LeakyReLU(0.1)(conv3_1_batchnorm1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# self.deconv2 = deconv(128, 64)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#     deconv2_deconv = tf.keras.layers.Conv2DTranspose(
#                 filters = 64,
#                 kernel_size = (4,4),
#                 padding = 'same',
#                 activation = tf.nn.relu,
#                 name = 'deconv2_deconv',
#                 strides = (2,2),
#                 use_bias = True)(conv3b_relu)#(conv3_1_relu)

# # self.conv2_1 = conv(self.batchNorm, 128, 64)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#     concat = tf.keras.layers.Concatenate(axis=3)([conv2b_relu, deconv2_deconv])
#     print("conv2b_relu, deconv2_deconv: ",conv2b_relu.shape, deconv2_deconv.shape, concat.shape)
#     conv2_1_conv = tf.keras.layers.Conv2D(
#                 filters = 64,
#                 kernel_size = (3,3),
#                 padding = 'same',
#                 activation = tf.nn.relu,
#                 name = 'conv2_1_conv',
#                 use_bias = False)(concat)
    
#     # Layer:       BatchNorm1
#     conv2_1_batchnorm1 = tf.keras.layers.BatchNormalization()(conv2_1_conv)

#     # Layer:       ReLU
#     conv2_1_relu = tf.keras.layers.LeakyReLU(0.1)(conv2_1_batchnorm1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# self.deconv1 = deconv(64, 32)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
    deconv1_deconv = tf.keras.layers.Conv2DTranspose(
                filters = 32,
                kernel_size = (4,4),
                padding = 'same',
                activation = tf.nn.relu,
                name = 'deconv1_deconv',
                strides = (2,2),
                use_bias = True)(conv2b_relu)#(conv2_1_relu)
    
# self.conv1_1 = conv(self.batchNorm, 64, 32)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
    concat = tf.keras.layers.Concatenate(axis=3)([conv1b_relu, deconv1_deconv])
    print("conv1b_relu, deconv1_deconv: ",conv1b_relu.shape, deconv1_deconv.shape, concat.shape)
    conv1_1_conv = tf.keras.layers.Conv2D(
                filters = 32,
                kernel_size = (3,3),
                padding = 'same',
                activation = tf.nn.relu,
                name = 'conv1_1_conv',
                use_bias = False)(concat)
    
    # Layer:       BatchNorm1
    conv1_1_batchnorm1 = tf.keras.layers.BatchNormalization()(conv1_1_conv)

    # Layer:       ReLU
    conv1_1_relu = tf.keras.layers.LeakyReLU(0.1)(conv1_1_batchnorm1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# self.deconv0 = deconv(32, 16)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
    deconv0_deconv = tf.keras.layers.Conv2DTranspose(
                filters = 16,
                kernel_size = (4,4),
                padding = 'same',
                activation = tf.nn.relu,
                name = 'deconv0_deconv',
                strides = (2,2),
                use_bias = True)(conv1_1_relu)

# self.conv0_1 = conv(self.batchNorm, 32 , 16)# ----------------------------------------------------------------------------------------------------------------------------------------------------------
    concat = tf.keras.layers.Concatenate(axis=3)([conv0b_relu, deconv0_deconv])
    conv0_1_conv = tf.keras.layers.Conv2D(
                filters = 16,
                kernel_size = (3,3),
                padding = 'same',
                activation = tf.nn.relu,
                name = 'conv0_1_conv',
                use_bias = False)(concat)
    
    # Layer:       BatchNorm1
    conv0_1_batchnorm1 = tf.keras.layers.BatchNormalization()(conv0_1_conv)

    # Layer:       ReLU
    conv0_1_relu = tf.keras.layers.LeakyReLU(0.1)(conv0_1_batchnorm1)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

# mask0 = self.pred_mask0(out_conv0_1) ----------------------------------------------------------------------------------------------------------------------------------------------------------
    pred_mask0_conv = tf.keras.layers.Conv2D(
                filters = 9,
                kernel_size = (3,3),
                padding = 'same',
                activation = tf.nn.softmax,
                name = 'pred_mask0_conv',
                use_bias = False)(conv0_1_relu)
    
    # Custom double-sigmoid activation function
    #pred_mask_0_activation = ConvolutionActivation()(pred_mask0_conv)

    # prob0 = self.softmax(mask0) ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # argmax = tf.keras.ops.argmax(pred_mask0_conv,axis = 3) #(b,y,x,label)
    #softmax = tf.keras.ops.argmax(axis = 1)(pred_mask0_conv) #(b,label,y,x)

    # #Local to Golbal layer

    sp_masks = SPLocalToGlobal((16,16),(128,128))(pred_mask0_conv)

    # #mask layer
    #sp_masks = SPMasks((4,4),(128,128))(local_to_global)

    # #Mean Features
    sp_feat = SPFeaturesMeanVectorized((16,16),(128,128),64)([conv0_1_relu,sp_masks])

    # #Classifcation
    conv_1d_classification = tf.keras.layers.Conv1D(
                filters = 16,
                kernel_size = 1,
                padding = 'same',
                activation = tf.nn.relu,
                name = 'conv_1d_classification',
                use_bias = False)(sp_feat)
    """
    conv_1d_classification_1 = tf.keras.layers.Conv1D(
                filters = 8,
                kernel_size = 1,
                padding = 'same',
                activation = tf.nn.relu,
                name = 'conv_1d_classification_1',
                use_bias = False)(conv_1d_classification)
    """
    conv_1d_classification_2 = tf.keras.layers.Conv1D(
                filters = 5,
                kernel_size = 1,
                padding = 'same',
                activation = tf.nn.softmax,
                name = 'conv_1d_classification_2',
                use_bias = False)(conv_1d_classification)
    
    # #Class Assignments
    class_assignments = SPClassAssignmentsVectorized((4,4),(128,128),5)([sp_masks,conv_1d_classification_2])
    
    # Max
    return tf.keras.Model(inputs = inputs, outputs = class_assignments)

