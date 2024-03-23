# read the file caption.json
# compute the similarity between the captions
import json
import numpy as np
from test import get_embedding_vector
data_from_json = {}
with open('caption.json') as f:
    data_from_json = json.load(f)

i = 0
data = []
data1 = []
data2 = []
data3 = []
for k,v in data_from_json.items():
    print("key: ", k)
    question = k.split("##")[0]
    rationale = v.split("##")[0]
    rationale1 = v.split("##")[1]
    print("value: ", v)
    embedding_vector_question = get_embedding_vector(question)
    embedding_vector_rationale = get_embedding_vector(rationale)
    embedding_vector_rationale1 = get_embedding_vector(rationale1)
    data1.append(embedding_vector_question)
    data2.append(embedding_vector_rationale)
    data3.append(embedding_vector_rationale1)
    # save the embedding vector to a file
    i += 1
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assuming 'data' is your dataset containing questions and rationales
# Preprocess your data as needed (e.g., using vectorization techniques)

# Convert it to a NumPy array
your_array1 = np.array(data1)
your_array2 = np.array(data2)
your_array3 = np.array(data3)
np.save("embedding_vector_question.npy", your_array1)
np.save("embedding_vector_rationale.npy", your_array2)
np.save("embedding_vector_rationale1.npy", your_array3)
# Now you can access the shape attribute
print("your_array1.shape:", your_array1.shape)
print("your_array2.shape:", your_array2.shape)
print("your_array3.shape:", your_array3.shape)
flattened_array1 = your_array1.reshape(10, -1)
flattened_array2 = your_array2.reshape(10, -1)
flattened_array3 = your_array3.reshape(10, -1)
# Now you can apply t-SNE with an appropriate perplexity value
# Perplexity should be less than the number of samples
tsne = TSNE(perplexity=5)  # Example perplexity value
transformed_data1 = tsne.fit_transform(flattened_array1)
transformed_data2 = tsne.fit_transform(flattened_array2)
transformed_data3 = tsne.fit_transform(flattened_array3)


plt.scatter(transformed_data1[:, 0], transformed_data1[:, 1], color='red')
plt.scatter(transformed_data2[:, 0], transformed_data2[:, 1], color='blue')
plt.scatter(transformed_data3[:, 0], transformed_data3[:, 1], color='green')
plt.colorbar()  # To show the color scale
plt.show()
#  sleep forever
while True:
    pass
