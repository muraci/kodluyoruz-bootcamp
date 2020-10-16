> ___Keşifsel Sonuçlar___

- Forex datasında ne olduğunu bilmediğim bağımsız değişkenlere göre hedef üzerinde regresyon tahmini, al-sat sınıflandırması ve kümeleme çalışması yapıldı. 

- Veri seti sınıfanmış bir davranış sergilediği görüldü. Target üzerinde yer alan pozitif, negatif ve 0 değerleri al-sat veya işlem yapma olarak nitelendirildi. 

- Veri setinde tek bir kategorik değişken bulunuyordu, bu değer dummy edildi ve ? işareti bulunan null değerler vardı ise drop sütun olarak drop edildi.

- Korelasyon matrixi inceledniğinde ufak kümelenmeler göze çarpıyordu, bunları kümeleyerek pca uygulaması yapıldı ve 3 farklı grupta pca birleştirmesi uygulandı. PCA uygulaması için %70 varyans beklendi.

- Kategorik eğilim gösteren değişkenler gözlemlendi, bu sütunlar üzerinde grafik ve p-değeri incelemesi yapıldı. İçerisinde nasıl bir bilgi barındırdığı bilinmediği için işlem yapılmadı.

- Keşifsel olarak target üzerinde pozitif-negatif bölünüp işlem yapılmaya çalışıldı. Zaman serisi olduğundan dolayı oldukça yanlış birdenemeydi. Harici target'in multak değeri alınarak negatif indirgenmesi sağlandı, target linear bir davranış sergilesede model üzerinde etkili sonuç alınamadı. 

- Yapılan analizler ve çıktılar neticesinde hedef değişkeninin regresyon için uygun olmadığını düşünüyorum, regresyon modelleri tamamen yöntemi uygulamak amacıyla yapıldı. Tahmin-gerçek grafik çıktıları ve r2_score değerleri çok beklenilen dışı sonuçlandığı için ödeve ve sunumlara eklenmedi.

> ___Regresyon Sonuçları___

- Gerekli varsayımlar incelendi, ama olumlu sonuçlar alınmadı.

- Zaman serisi olmasından kaynaklı shuffle parametresi false olarak kullanıldı, bu düzeltme cross değerleri ve grid ayarlaması içinde TimeSeriesSplit fonsiyonu kullanılarak uygulandı, ama beklenen dışı sonuçlar verdiği için sunumda bulunmuyor.

- Model dokunulmamış, 8 değikenli pca ve kümelenmiş pca üzerinden 3 farklı dataya uygulandı. En iyi çıktı 8 değişkenli pca'dan alındı. Ayrıca hızlı sonuç alabilmek adına bu data tercih edildi.

<p align="center">
<img src="https://github.com/Kodluyoruz-Ankara-Veri-Bilimi/muratacikgoz/blob/master/img/forex_table.PNG"/></p>

- Gerekli parametre ayarlamaları yapılarak karşılaştırma tablosu oluşturuldu. En iyi çıktı RMSE değerine lightgbm modelinden alındı.

> ___Sınıflandırma Sonuçları___

- Al-Sat veya işlem yapma olarak yeni bir sütun oluşturuldu. Zaman serisi kaynaklı shuffle false yapılarak işlemlere devam edildi.

- Veride işlem yapma (0) değerleri 70/6000 az olmasından dolayı, öğrenmesi çok başarılı olmadı. 

<p align="center">
<img src="https://github.com/Kodluyoruz-Ankara-Veri-Bilimi/muratacikgoz/blob/master/img/forex_cla.PNG"/></p>

- Al-sat değişkeninin pozitif mi, negatif mi olduğu konusunda nitelendirme yapılamadığı için maliyet analizi konusunda yorum yapılamadı, ama genel tabloya bakılırsa %50 oranında bir başarı yakalnamış görünüyor.

> ___Kümeleme Sonuçları___

- PCA uygulanmış modeller tercih edildi. PCA-1, 8 değişkenli olan veri seti, PCA-2 ise kümelenmiş pca uygulanan veri setidir.

<p align="center">
<img src="https://github.com/Kodluyoruz-Ankara-Veri-Bilimi/muratacikgoz/blob/master/img/forex_pca.PNG"/></p>

- İşlem yapma (0) değerlerinin az olmasından kaynaklı -70 tane- çok azınlıkta yeşil olarak görülmektedirler.

- İki pca modelinin karşılaştırması yapıldı, 8 değişkenli olandan daha iyi kümelenme gözlemlendi.

- Grafikler 5 aralığında bölünme olduğunu söyleselerde, sınıflandırma karşılaştırması için 3 tercih edildi. Ama bu aralıkta oldukça iyi kümelenme görülmektedir.

<p align="center">
<img src="https://github.com/Kodluyoruz-Ankara-Veri-Bilimi/muratacikgoz/blob/master/img/forex_nclu.PNG"/></p>

- İki farklı grafikte renk değişkenleri olarak, kümeleme etiketleri ve sınıflandırma grupları kullanılmıştır. Benzer bir davranış sergilemedikleri görülmektedir.

<p align="center">
<img src="https://github.com/Kodluyoruz-Ankara-Veri-Bilimi/muratacikgoz/blob/master/img/fores_cluvscla.PNG"/></p>

- Ağağıdaki grafikte sınıflandırma ve kümeleme algoritmalarını karşılaştırmak için, sınıflandırmdan gelen gruplara renk -mavi,kırmızı,yeşil- atandı, kümelemeden gelen etiketlere text -0,1,2- atandı. Farklılıkları görmek adına büyük grafiği çalışma dosyasından inceleyebilirsiniz.

<p align="center">
<img src="https://github.com/Kodluyoruz-Ankara-Veri-Bilimi/muratacikgoz/blob/master/img/forex_clutext.PNG"/></p>
