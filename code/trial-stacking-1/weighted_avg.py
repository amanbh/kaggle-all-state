def mae_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.sum(weights * l1_x_train, axis=1)
    return mean_absolute_error(y_train, final_prediction)


starting_values = np.random.uniform(size=l1_x_train.shape[1])

