> Keşifsel Sonuçlar

- Veri setinde çıktı olarak, banka müşterilerine kredi sağlayabilmek için kişinin demografisini incelenip, müşteri hakkında iyi veya kötü olarak yorum yapılması beklenmektedir.

- Veriyi keşifsel olarak analiz ettiğimizde, çok fazla null değer görünmekte, bunlar yaklaşık datanın 3te1'ini oluşturmaktadır. Bunları drop etmek yerine, ortalama değer ötesinde bağıntılar kurarak dolduruldu. Bu dordurma sayesinde modelde daha iyi öğrenme yakalandığı söylenebilir.

- Kategorik eğilim gösteren 3 sutun detaylı olarak incelendi; dağılımları, p-değerleri hem numerik hem de kategorik olarak hedef üzerindeki etkisi araştırıldı. Ama ordinal bi yapıda oldukları için manipule edilmedi.

- Veri setinde çoğu değişken için histogram grafiklerinde aykırı değerlerden dolayı çarpıklıklar ve normal dağılımlı bir şekil gözlemlendi. Ama qqplot ve shapiro testinde normal dağılım sergilemediği sonucu çıktı.

- Bazı kategoriler için 0 değeri, müşteri kaydının olmadığı anlamına gelmektedir. Bu sütunlar için detaylı inceleme yapıdı. Ayrıca anlamsız aykırı değerler düzeltildi, 100 yıllık müşteri verisi görülüyordu.

- P-değerlerine göre hiç bir değişken normal dağılmıyordu, hedef üzerinde tek bir değişkenin anlamlı bir farklılığı yoktu. Kredinin onaylanabilmesi için diğer değerler anlamlı-belirleyici görünüyordu.

> Model Sonuçları

- Modele uygulayabilmek için dönüşümlerin yapıldığı, drop değerler inolduğu 6 farklı dataseti oluşturuldu. Model çıktıları karşılaştırılarak, bu setler arasından scale edilmiş olan tercih edildi. Diğer modeller bu set üzerine kuruldu. Verinin scale edilmiş olması zamansal olarak kazanım sağladığı için tercih edilidiğide söylenebilir.

- Veri üzerinde cross farklılıklarını görmek için split edilen score değerleri grafikleştirildi. Bu farklıkların ortalaması sayesinde overfiti önlemek için iyi bir seçenek sunuyor.
  
<p align="center">
<img src="https://github.com/Kodluyoruz-Ankara-Veri-Bilimi/muratacikgoz/blob/master/img/hmelq_table.PNG"/></p>

- Modellere en iyi çıktı alabileceğimiz parametreler uygulanarak, karşılaştırmak için tablo ve grafik oluşturuldu. Bu tabloya göre en iyi açıklayıcılık lightgbm modelinden alındı, bazı modellerde overfit yaşandı, ama cross değerleri incelendiğinde farklılık gözlemlendi.

<p align="center">
<img src="https://github.com/Kodluyoruz-Ankara-Veri-Bilimi/muratacikgoz/blob/master/img/hmelq_auc.PNG"/></p>

- Banka verisi olmasından kaynaklı, kötü müşterileri doğru tahmin etmemiz gerekmektedir. Harici durumda banka için maliyetli olabilir, bundan dolayı recall değeri daha öncelikli gözlemlendi. En iyi sonucu yine lightgbm verdi.

<p align="center">
<img src="https://github.com/Kodluyoruz-Ankara-Veri-Bilimi/muratacikgoz/blob/master/img/hmelq_matrix.PNG"/></p>

- Lightgbm modeli üzerinden değişkenlerin önem incelemesi yapıldı. Meslek grupları diğer deomgrafik özelliklere göre çok düşük davranış sergiledi, malvarlıkları burada model üzerinde önemli bir etkiye sahiptir sonucu çıktı.

<p align="center">
<img src="https://github.com/Kodluyoruz-Ankara-Veri-Bilimi/muratacikgoz/blob/master/img/hmelq_imp.PNG"/></p>


