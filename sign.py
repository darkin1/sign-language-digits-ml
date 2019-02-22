import tensorflow as tf

model = tf.keras.models.load_model('./models/sign-10-40-epochs-1550868468')

### Predict answer
key = 1
pX = X[key][np.newaxis,:,:,:] # that same as: array([X[1]])

predicted = model.predict(pX)
print(predicted)

# predicted_proba = model.predict_proba(pX)
# print(predicted_proba)

predicted_class_number = model.predict_classes(pX)
print(predicted_class_number)

print("Classes probability=%s, The best predicted number=%s, Y=%s" % (predicted, predicted_class_number, y[key]))

print(np.round(predicted_class_number, 1))


# Show predicted image
plt.imshow(np.squeeze(X[key]), cmap="gray")
plt.show()