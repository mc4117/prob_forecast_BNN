import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers

import numpy as np

import climetlab as cml
import mars_surf_temp as mst

step = 3
lead_time = 24*step

# set up climetlab data object
ds_op = cml.load_dataset('mars-surf-temp', datetime_init = '2016-01-01', datetime_end = '2019-12-30', forecast_type = ['op'], grid_res = '1/1', step = lead_time, extra_vars = ['soil_temp', 'sea_ice', 'snow_depth'], extra_step = [24*1])

# read in state-independent variables
land_mask = cml.load_source('file', '/perm/momc/mars_data/land_sea_mask_1.grib').to_xarray().isel(step = 0).isel(surface = 0)
topo = cml.load_source('file', '/perm/momc/mars_data/topo_1.grib').to_xarray().isel(step = 0).isel(surface = 0)
koeppen = cml.load_source('file', '/perm/momc/mars_data/koeppen_1.nc').to_xarray()
leaf = cml.load_source('file', '/perm/momc/mars_data/leaf_one_day_1.nc').to_xarray()

# define size of the domain
lat_isel_init = 0
lat_isel_end = 181
lon_isel_init = 0
lon_isel_end = 360

# number of days used in sliding window approach
sliding_bias_num = 30

# make training data generator choosing predictors and date ranges
dg_train_op = mst.DataGenerator(ds_op, 't2m', 'op_forecast', selection_type = 'date_time',
                                lat_isel_init = lat_isel_init, lat_isel_end = lat_isel_end, lon_isel_init = lon_isel_init, lon_isel_end = lon_isel_end,
                                datetime_init = '2016-01-01', datetime_end = '2017-12-31', grid_approach = True, datetime_feature = True, sliding_bias_num = sliding_bias_num, mean_bias = 0, std_bias = 1,
                                batch_size = 8, load = True, const_var = [land_mask, topo, koeppen, leaf], shuffle = True, latitude_feature = True, 
                                longitude_feature = False, zenith_angle = True, model_change = False, categorical = True, one_hot = True, equi_bins = True, bin_number = 150)
# make validation data generator choosing predictors and date ranges
dg_valid_op = mst.DataGenerator(ds_op, 't2m', 'op_forecast', selection_type = 'date_time', datetime_feature = True,
                               lat_isel_init = lat_isel_init, lat_isel_end = lat_isel_end, lon_isel_init = lon_isel_init, lon_isel_end = lon_isel_end,
                               datetime_init = '2018-01-01', datetime_end = '2018-12-31',
                               batch_size = 8, load = True, mean = dg_train_op.mean, sliding_bias_num = sliding_bias_num,
                               std = dg_train_op.std, mean_bias = 0, std_bias = 1,
                               grid_approach = True, const_var = [land_mask, topo, koeppen, leaf], shuffle = True, equi_bins = True,
                               latitude_feature = True, longitude_feature = False, zenith_angle = True, model_change = False, categorical = True, one_hot = True, bins = dg_train_op.bins, bin_number = 150)
# make test data generator choosing predictors and date ranges
dg_test_op = mst.DataGenerator(ds_op, 't2m', 'op_forecast', selection_type = 'date_time', datetime_feature = True, sliding_bias_num = sliding_bias_num,
                               lat_isel_init = lat_isel_init, lat_isel_end = lat_isel_end, lon_isel_init = lon_isel_init, lon_isel_end = lon_isel_end,
                               datetime_init = '2019-01-01', datetime_end = '2019-12-20', grid_approach = True, equi_bins = True,
                               batch_size = 8, load = True, mean = dg_train_op.mean, std = dg_train_op.std, mean_bias = 0, std_bias = 1,
                               const_var = [land_mask, topo, koeppen, leaf], shuffle = False, latitude_feature = True, longitude_feature = False, zenith_angle = True,
                               model_change = False, categorical = True, one_hot = True, bins = dg_train_op.bins, bin_number = 150)


def build_mlp(input_shape):
    # function which builds neural network
    x = input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(100)(x)
    x = tf.keras.layers.Activation('ReLU')(x)

    for i in range(3):
        y = x = tf.keras.layers.Dense(units=100)(x)
        x = tf.keras.layers.Activation('ReLU')(x)
        # skip connection which enable deeper neural networks
        x = tf.keras.layers.Add()([y, x])

    x = tf.keras.layers.Dense(units=100)(x)
    x = tf.keras.layers.Activation('ReLU')(x)

    x = tf.keras.layers.Dense(150, activation = None)(x)
    # output layer predicts a probabilistic distribution rather than a deterministic output
    output = tfpl.OneHotCategorical(150, convert_to_tensor_fn = tfd.Distribution.mean)(x)

    return tf.keras.models.Model(input, output)

# calculate cos-latitude weights for the loss function due to the use of a lat-lon grid
weights = np.cos(np.deg2rad(dg_test_op.total_data.latitude))
weights /= weights.mean()
weights_expand = np.expand_dims(weights.data, axis = 1)
weights_list = [weights_expand for i in range(360)]
weights_concat = np.concatenate(weights_list, axis = 1)
weights_data = tf.convert_to_tensor(weights_concat, dtype = tf.float32)


def nll(y_true, y_pred):
    # negative loglikelihood function used to calculate the loss of the distribution
    delta = -y_pred.log_prob(y_true)
    return tf.reshape(delta, [-1, 181, 360, 1]) * tf.reshape(weights_data, [-1, 181, 360, 1])

# build BNN
mlp = build_mlp((16,))

mlp.compile(loss=nll,
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])

print(mlp.summary())

# callbacks used in training
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            patience=5,
            factor=0.5,
            verbose=1)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=12,
                        verbose=1,
                        mode='auto'
                    )

# train BNN
mlp.fit(dg_train_op, epochs=200, validation_data=dg_valid_op,
          callbacks=[reduce_lr_callback, early_stopping_callback]
         )


# calculate new forecast and new errors
forecast_unstandardise = dg_test_op.forecast.isel(time = slice(2*sliding_bias_num, None))
bins_centre = dg_test_op.bins
std_bias = dg_test_op.std_bias
mean_bias = dg_test_op.mean_bias
preds = (np.dot(preds_new, bins_centre).reshape(forecast_unstandardise.shape[1:]) * std_bias) + mean_bias
new_forecast = - preds.reshape(forecast_unstandardise.shape[1:]) + forecast_unstandardise
analysis = dg_test_op.analysis_combined.isel(time = slice(2*sliding_bias_num,None))
err_orig = (analysis - forecast_unstandardise)**2
print('orig')
print(np.sqrt(err_orig.mean().values))
print(np.sqrt((err_orig.sel(number = 0)*weights_concat).mean().values))
err = (analysis - new_forecast)**2
print('new')
print(np.sqrt(err.mean().values))
print(np.sqrt((err[:, :, :, 0]*weights_concat).mean().values))
