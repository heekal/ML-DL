# 🚀 ULTIMATE GUIDE: Grokking Deep Learning! 🚀

> **"Anyone can read this book and understand how deep learning really works."**  
> Companion guide berbasis buku **Grokking Deep Learning** karya **Andrew W. Trask** (Published by Manning Publications). Repository ini berisi recreate materi tiap chapter dalam bentuk Google Colab yang siap dijalankan.

---

## 🌟 Why This Will Change Your Deep Learning Journey

Buku ini dirancang sebagai jalur masuk paling rendah untuk memahami deep learning dari dasar. Fokusnya bukan sekadar memakai library tingkat tinggi, tetapi memahami apa yang terjadi di balik neural network: prediction, error, gradient descent, backpropagation, regularization, convolution, NLP, RNN, LSTM, framework mini, sampai federated learning.

Materi di repository ini dibuat ulang sebagai notebook Colab agar setiap konsep bisa dibaca, dijalankan, dan diuji langsung.

## 🛠️ The Unstoppable Tech Stack

Stack utama yang digunakan sepanjang perjalanan ini:

* **Python:** Bahasa utama untuk membangun semua eksperimen deep learning.
* **NumPy:** Core numerical engine untuk memahami matrix, vector, dot product, dan neural network dari nol.
* **Google Colab:** Environment interaktif untuk menjalankan notebook tanpa setup lokal berat.
* **TensorFlow/Keras Dataset Utilities:** Digunakan untuk mengambil dataset seperti MNIST pada beberapa chapter.
* **Custom Mini Deep Learning Framework:** Framework sederhana buatan sendiri berisi `Tensor`, autograd, optimizer, layers, embedding, cross-entropy, RNN, dan LSTM.
* **NLP & Sequence Modeling Utilities:** Dipakai untuk word embeddings, character modeling, RNN, dan LSTM.
* **Privacy-Preserving Learning Tools:** Termasuk federated learning dan homomorphic encryption menggunakan `phe`.

---

## 📚 Epic Chapter Breakdown

| Chapter | Epic Title | What You Will Master |
| --- | --- | --- |
| **1** | Introducing Deep Learning | Memahami kenapa deep learning penting, apa yang dibutuhkan untuk mulai, dan kenapa buku ini fokus pada pemahaman dari dasar. |
| **2** | Fundamental Concepts | Membedakan AI, machine learning, deep learning, supervised/unsupervised learning, serta parametric/nonparametric models. |
| **3** | Neural Prediction | Membuat neural network pertama, forward propagation, weighted sum, multiple inputs, multiple outputs, dan NumPy basics. |
| **4** | Gradient Descent | Memahami compare, error, hot-and-cold learning, derivative, gradient descent, alpha, divergence, dan weight update. |
| **5** | Generalizing Gradient Descent | Melatih banyak weights sekaligus, multiple inputs/outputs, freezing weight, dan memahami dot product sebagai similarity. |
| **6** | Backpropagation | Membangun deep neural network pertama, streetlight problem, stochastic gradient descent, correlation, relu, dan backpropagation. |
| **7** | Visualizing Neural Networks | Menyederhanakan cara melihat neural network melalui layers, matrices, vectors, algebra, dan visual architecture. |
| **8** | Regularization & Batching | Menghadapi overfitting dengan early stopping dan dropout, serta mempercepat training dengan batch gradient descent. |
| **9** | Activation Functions | Memahami sigmoid, tanh, relu, softmax, output probabilities, derivative activation, dan upgrade MNIST network. |
| **10** | Convolutional Neural Networks | Mengenal convolution, weight sharing, kernels, pooling, dan CNN sederhana berbasis NumPy. |
| **11** | Neural Networks for Language | Masuk ke NLP, one-hot encoding, word embeddings, sentiment classification, fill-in-the-blank, dan word analogies. |
| **12** | Recurrent Neural Networks | Memproses variable-length data, sentence vectors, identity matrix, recurrent transition, dan backpropagation through sequence. |
| **13** | Building a Deep Learning Framework | Membangun mini framework sendiri: Tensor, autograd, SGD, Linear, Sequential, Embedding, CrossEntropyLoss, dan RNNCell. |
| **14** | Long Short-Term Memory | Membuat character language model ala Shakespeare, truncated backpropagation, vanishing/exploding gradients, dan LSTMCell. |
| **15** | Federated Learning | Memahami privacy problem, federated learning, spam detection, secure aggregation, dan homomorphic encryption. |
| **16** | Where to Go from Here | Menentukan langkah lanjut: PyTorch, course lain, textbook matematis, blog, paper implementation, GPU, open source, dan komunitas. |

---

## 📁 File Structure

```text
GrokkingDeepLearning/
├── introducing_deep_learning_why_you_should_learn_it_colab.ipynb
├── fundamental_concepts_how_do_machines_learn_colab.ipynb
├── introduction_to_neural_prediction_forward_propagation_colab.ipynb
├── introduction_to_neural_learning_gradient_descent_colab.ipynb
├── learning_multiple_weights_at_a_time_generalizing_gradient_descent_colab.ipynb
├── building_your_first_deep_neural_network_introduction_to_backpropagation_colab.ipynb
├── how_to_picture_neural_networks_in_your_head_and_on_paper_colab.ipynb
├── learning_signal_and_ignoring_noise_introduction_to_regularization_and_batching_colab.ipynb
├── modeling_probabilities_and_nonlinearities_activation_functions_colab.ipynb
├── neural_learning_about_edges_and_corners_intro_to_convolutional_neural_networks_colab.ipynb
├── neural_networks_that_understand_language_king_man_woman_colab.ipynb
├── neural_networks_that_write_like_shakespeare_recurrent_layers_for_variable_length_data_colab.ipynb
├── introducing_automatic_optimization_lets_build_a_deep_learning_framework_colab.ipynb
├── learning_to_write_like_shakespeare_long_short_term_memory_colab.ipynb
├── deep_learning_on_unseen_data_introducing_federated_learning_colab.ipynb
├── where_to_go_from_here_a_brief_guide_colab.ipynb
└── README.md
```

---

## 🎯 Who Is This For?

Repository ini cocok untuk:

* Mahasiswa yang ingin memahami deep learning dari nol.
* Pembelajar Python yang ingin tahu cara neural network bekerja di balik framework.
* Calon AI Engineer, ML Engineer, Research Engineer, atau Data Scientist.
* Siapa pun yang ingin belajar deep learning tanpa langsung tenggelam dalam notasi matematika tingkat lanjut.
* Pembaca yang ingin menjalankan ulang materi buku dalam format Google Colab.

Tidak wajib menguasai linear algebra, calculus, convex optimization, atau machine learning sebelumnya. Cukup nyaman dengan Python dasar dan matematika SMA.

---

## ⚖️ Copyright & Acknowledgments

*Based on **Grokking Deep Learning** by **Andrew W. Trask**. Copyright © 2019 Manning Publications Co. All rights reserved. Published by Manning Publications.*

Repository ini dibuat sebagai companion learning material dalam format Google Colab. Semua hak cipta materi asli tetap milik penulis dan penerbit.
