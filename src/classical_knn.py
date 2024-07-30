from combine_color_texture_features import combine_color_texture_features
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
import numpy as np

def size(path) -> int:
    return len(os.listdir(path))

# num_train_imgs is even; k is the number of nearest neighbors
def run_knn(num_train_imgs: int, cat_1: str, cat_2: str, k: int = 5):

    cat_1_path = os.path.join(os.getcwd(), "tests", "test_images_full", cat_1)
    cat_2_path = os.path.join(os.getcwd(), "tests", "test_images_full", cat_2)

    #choose indices for target vectors
    train_cat_1 = np.random.choice(list(range(size(cat_1_path))), num_train_imgs // 2, replace=False)
    train_cat_2 = np.random.choice(list(range(size(cat_2_path))), num_train_imgs // 2, replace=False)
    test_cat_1 = np.array([x for x in range(size(cat_1_path)) if x not in train_cat_1])
    test_cat_2 = np.array([y for y in range(size(cat_2_path)) if y not in train_cat_2])

    X_train = np.empty(shape=(num_train_imgs, 80))
    Y_train = np.empty(shape=(num_train_imgs,))

    j = 0
    for i in train_cat_1:
        X_train[j] = combine_color_texture_features(os.path.join(cat_1_path, cat_1 + f"_{i}.jpg"))
        Y_train[j] = 0
        j += 1

    for i in train_cat_2:
        X_train[j] = combine_color_texture_features(os.path.join(cat_2_path, cat_2 + f"_{i}.jpg"))
        Y_train[j] = 1
        j += 1

    
    num_test_imgs = size(cat_1_path) + size(cat_2_path) - num_train_imgs

    X_test = np.empty(shape=(num_test_imgs, 80))
    Y_test = np.empty(shape=(num_test_imgs,))



    j = 0
    for i in test_cat_1:
        X_test[j] = combine_color_texture_features(os.path.join(cat_1_path, cat_1 + f"_{i}.jpg"))
        Y_test[j] = 0
        j += 1

    for i in test_cat_2:
        X_test[j] = combine_color_texture_features(os.path.join(cat_2_path, cat_2 + f"_{i}.jpg"))
        Y_test[j] = 1
        j += 1

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, pred)
    print(f"Categories: {cat_1} and {cat_2}")
    print("Accuracy:", accuracy)
    return accuracy


cats = os.listdir(os.path.join(os.getcwd(), "tests", "test_images_full"))

def run_all():
    finished = []
    X = []
    Y = []
    for i in cats:
        for j in cats:
            if not i == j and ((i, j) not in finished) and ((j, i) not in finished):
                accuracy = run_knn(10, i, j)
                X.append(f"{i} & {j}")
                Y.append(accuracy * 100)
                finished.append((i, j))

    fig, ax = plt.subplots(figsize =(16, 9))
    ax.barh(X, Y)

    plt.xlabel("Categories")
    plt.ylabel("Accuracy (%)")

    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)


    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)

    # Show top values 
    ax.invert_yaxis()

    ax.set_title('Accuracy of binary classification tasks (Classical KNN)',
                loc ='center')

def run_one(reps, cat_1, cat_2):
    accs = []
    for _ in range(reps):
        accuracy = run_knn(10, cat_1, cat_2)
        accs.append(accuracy * 100)
    print(f"Average: {sum(accs) / len(accs)}")

run_one(3, "airplane", "beaver")




