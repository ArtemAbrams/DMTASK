import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Завантажуємо дані з CSV-файлу
data = pd.read_csv('./customer_features.csv')
X = data.values.astype(float)

# Функція косинусної відстані
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Реалізація K‑Means з косинусною відстанню
def kmeans_cosine(X, k, max_iter=100, tol=1e-4):
    rng = np.random.RandomState(42)
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(max_iter):
        distances = np.array([[cosine_distance(x, c) for c in centroids] for x in X])
        labels = distances.argmin(axis=1)
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    distortion = sum(cosine_distance(X[i], centroids[labels[i]]) for i in range(len(X)))
    return labels, centroids, distortion

if __name__ == "__main__":
    ks = range(1, 7)
    distortions = []
    print("Elbow method — distortion для різних k:")
    for k in ks:
        _, _, d = kmeans_cosine(X, k)
        distortions.append(d)
        print(f"  k = {k}: distortion = {d:.6f}")

    # Точку “зламу” (elbow) на графіку видно при переході від k = 2 до k = 3:
    # після трьох кластерів додавання ще одного дає незначне покращення, тому optimal_k = 3
    optimal_k = 3
    labels, centroids, _ = kmeans_cosine(X, optimal_k)

    print(f"\nВибране k = {optimal_k}")
    for idx, (row, lbl) in enumerate(zip(data.values, labels), start=1):
        print(f"  Запис #{idx} (age={row[0]}, income={row[1]}, score={row[2]}) → кластер {lbl}")

    print("\nЦентроїди кластерів:")
    for j, c in enumerate(centroids):
        print(f"  Кластер {j}: age={c[0]:.2f}, income={c[1]:.2f}, score={c[2]:.2f}")

    # 4) Побудова графіка Elbow
    plt.figure(figsize=(8, 5))
    plt.plot(ks, distortions, marker='o')
    plt.xlabel('Кількість кластерів k')
    plt.ylabel('Сумарна косинусна відстань (distortion)')
    plt.title('Elbow Method для вибору оптимального k')
    plt.xticks(ks)
    plt.grid(True)
    plt.show()
