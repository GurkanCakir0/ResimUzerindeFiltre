import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

def OrtalamaFiltre():
    print("Ortalama Filtre Seçildi")
    # Görüntüyü yükle ve gri tonlamaya dönüştür
    image = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)

    # Box (Ortalama) Filtresi uygula (3x3 kernel)
    box_filtered=cv2.blur(image, (3,3))  # OpenCV'de hazır fonksiyon

    # Görüntüleri karşılaştırmalı göster
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image, cmap='gray')

    axes[1].imshow(box_filtered, cmap='gray')
    plt.show()


def MedianFiltre():
    print("Median Filtre Seçildi")
    # Görüntüyü yükle ve gri tonlamaya dönüştür
    image = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)

    # Medyan filtreyi uygula (3x3, 5x5 ve 7x7 şablonlar)
    median_filtered_3 = cv2.medianBlur(image, 3)
    median_filtered_5 = cv2.medianBlur(image, 5)
    median_filtered_7 = cv2.medianBlur(image, 7)


    # Görüntüleri karşılaştırmalı olarak göster
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')

    axes[1].imshow(median_filtered_3, cmap='gray')
    axes[1].set_title('Medyan Filtre (3x3)')
    axes[1].axis('off')

    axes[2].imshow(median_filtered_5, cmap='gray')
    axes[2].set_title('Medyan Filtre (5x5)')
    axes[2].axis('off')

    axes[3].imshow(median_filtered_7, cmap='gray')
    axes[3].set_title('Medyan Filtre (7x7)')
    axes[3].axis('off')
    plt.show()


def GaussFiltre():
    print("Gauss Filtre Seçildi")
    # Görüntüyü yükle ve gri tonlamaya dönüştür
    image = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)

    # Farklı boyutlarda Gauss filtresi uygula
    gaussian_filtered_3 = cv2.GaussianBlur(image, (3, 3), 1)  # 3x3 kernel, σ=1
    gaussian_filtered_5 = cv2.GaussianBlur(image, (5, 5), 2)  # 5x5 kernel, σ=2
    gaussian_filtered_7 = cv2.GaussianBlur(image, (7, 7), 3)  # 7x7 kernel, σ=3

    # Görüntüleri karşılaştırmalı olarak göster
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')

    axes[1].imshow(gaussian_filtered_3, cmap='gray')
    axes[1].set_title('Gaussian Filtre (3x3, σ=1)')
    axes[1].axis('off')

    axes[2].imshow(gaussian_filtered_5, cmap='gray')
    axes[2].set_title('Gaussian Filtre (5x5, σ=2)')
    axes[2].axis('off')

    axes[3].imshow(gaussian_filtered_7, cmap='gray')
    axes[3].set_title('Gaussian Filtre (7x7, σ=3)')
    axes[3].axis('off')
    plt.show()


def SobelFiltre():
    print("Sobel Filtre Seçildi")
    # Görüntüyü yükle ve gri tonlamaya dönüştür
    image = cv2.imread('gorsel.jpg', cv2.IMREAD_GRAYSCALE)

    # Sobel filtresi uygulanarak kenar tespiti
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    sobel_edges = np.uint8(np.clip(sobel_edges, 0, 255))

    # Prewitt filtresi uygulanarak kenar tespiti (manuel)
    prewitt_x = cv2.filter2D(image, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
    prewitt_y = cv2.filter2D(image, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
    prewitt_edges = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
    prewitt_edges = np.uint8(np.clip(prewitt_edges, 0, 255))

    # Laplacian filtresi uygulanarak kenar tespiti
    laplacian_edges = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    laplacian_edges = np.uint8(np.clip(np.abs(laplacian_edges), 0, 255))

    # Keskinleştirilmiş görüntüleri hesapla
    alpha = 1.0  # Ağırlık faktörü
    sharp_sobel = cv2.addWeighted(image, 1, sobel_edges, alpha, 0)
    sharp_prewitt = cv2.addWeighted(image, 1, prewitt_edges, alpha, 0)
    sharp_laplacian = cv2.addWeighted(image, 1, laplacian_edges, alpha, 0)

    # Görüntüleri karşılaştırmalı olarak göster
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Orijinal Görüntü')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(sobel_edges, cmap='gray')
    axes[0, 1].set_title('Sobel Kenarları')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(prewitt_edges, cmap='gray')
    axes[0, 2].set_title('Prewitt Kenarları')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(laplacian_edges, cmap='gray')
    axes[0, 3].set_title('Laplacian Kenarları')
    axes[0, 3].axis('off')

    axes[1, 1].imshow(sharp_sobel, cmap='gray')
    axes[1, 1].set_title('Keskinleştirilmiş (Sobel)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(sharp_prewitt, cmap='gray')
    axes[1, 2].set_title('Keskinleştirilmiş (Prewitt)')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(sharp_laplacian, cmap='gray')
    axes[1, 3].set_title('Keskinleştirilmiş (Laplacian)')
    axes[1, 3].axis('off')
    plt.show()


def PrewittFiltre():
    print("Prewitt Filtre Seçildi")
    # Görüntüyü yükle ve gri tonlamaya dönüştür
    image = cv2.imread('gorsel4.jpg', cv2.IMREAD_GRAYSCALE)

    # Prewitt Filtresi Çekirdekleri
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)  # Yatay değişim (Dikey kenarlar)

    prewitt_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=np.float32)  # Dikey değişim (Yatay kenarlar)

    # Prewitt Filtresini Uygula
    prewitt_x_filtered = cv2.filter2D(image, -1, prewitt_x)  # Dikey kenarlar
    prewitt_y_filtered = cv2.filter2D(image, -1, prewitt_y)  # Yatay kenarlar

    # X ve Y yönündeki kenarları birleştir
    prewitt_combined = cv2.addWeighted(prewitt_x_filtered, 0.5, prewitt_y_filtered, 0.5, 0)

    # Görüntüleri karşılaştırmalı olarak göster
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')

    axes[1].imshow(prewitt_x_filtered, cmap='gray')
    axes[1].set_title('Prewitt-X (Dikey Kenarlar)')
    axes[1].axis('off')

    axes[2].imshow(prewitt_y_filtered, cmap='gray')
    axes[2].set_title('Prewitt-Y (Yatay Kenarlar)')
    axes[2].axis('off')

    axes[3].imshow(prewitt_combined, cmap='gray')
    axes[3].set_title('Birleşik Prewitt Kenarları')
    axes[3].axis('off')

    plt.show()


def LaplacianFiltre():
    print("Laplacian Filtre Seçildi")
    # Görüntüyü yükle ve gri tonlamaya dönüştür
    image = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))

    laplacian_edges = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    laplacian_edges = np.uint8(np.clip(np.abs(laplacian_edges), 0, 255))

    # Görüntüleri gösterme
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')

    axes[1].imshow(laplacian_edges, cmap='gray')
    axes[1].set_title('Zero Padding')
    axes[1].axis('off')

    axes[2].imshow(laplacian_edges, cmap='gray')
    axes[2].set_title('Replicate Padding')
    axes[2].axis('off')
    plt.show()


def FaceBlurFiltre():
    print("FaceBlur Filtre Seçildi")
    # Görüntüyü yükle ve gri tonlamaya dönüştür
    image = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))

    # 3x3 Ortalama (Box) Filtresi
    kernel = np.ones((3, 3), np.float32) / 9

    # Farklı padding yöntemleriyle filtreleme
    valid_filtered = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)  # Zero padding
    replicate_filtered = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)  # Replicate padding

    # Görüntüleri gösterme
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')

    axes[1].imshow(valid_filtered, cmap='gray')
    axes[1].set_title('Zero Padding')
    axes[1].axis('off')

    axes[2].imshow(replicate_filtered, cmap='gray')
    axes[2].set_title('Replicate Padding')
    axes[2].axis('off')

    plt.show()


def menu():
    FiltreList=[OrtalamaFiltre,MedianFiltre,GaussFiltre,SobelFiltre,PrewittFiltre,LaplacianFiltre,FaceBlurFiltre]
    while True:
        print("1-Ortalama Filtre")
        print("2-Median Filtre")
        print("3-Gauss Filtre")
        print("4-Sobel Filtre")
        print("5-Prewitt Filtre")
        print("6-Laplacian Filtre")
        print("7-FaceBlur Filtre")
        print("0-Programdan Çık")
        seciminiz=int(input("Lütfen Yapmak istediğiniz İşlemi Seçiniz (0-7):"))
        print("\n" * 40)
        if seciminiz <=7 and seciminiz >=1:
            FiltreList[seciminiz-1]()

        elif seciminiz==0:
            print("Çıkış Yapılıyor")
            break
        else:
            print("Lütfen Seçim için 0 ile 7 arasında bir sayı seçiniz! "),


menu()