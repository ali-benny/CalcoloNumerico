Relazione Progetto di Calcolo Numerico

Benatti Alice, Manuelli Matteo, Qayyum Shahbaz Ali 

*Alma Mater Studiorum - Università di Bologna* 

### 1.Introduzione

Per svolgere il progetto si farà uso dei moduli `numpy , skimage e matplotlib` utilizzando il linguaggio Python. Il progetto ha come scopo quello di comprendere e mettere in atto metodi per ricostruire immagini blurrate e svolgere il lavoro opposto, quindi generare immagini corrotte (dal rumore) a partire da un immagine originale. 

Il problema che ci è stato presentato riguarda la ricostruzione di immagini corrotte attraverso il blur Gaussiano. Verrà analizzata inizialmente l’immagine `data.camera()` importata da `skimage` , successivamente verranno analizzate un set di 8 immagini con oggetti geometrici di colore uniforme su sfondo nero, realizzate da noi. Il problema di deblur consiste nella ricostruzione di un immagine a partire da un dato acquisito mediante il seguente modello:

<div style="text-align:center">b = Ax + η</div>

Dove b rappresenta l’immagine corrotta, x l’immagine originale che vogliamo ricostruire, A l’operatore che applica il blur Gaussiano ed η il rumore additivo con distribuzione Gaussiana di media ⊬ e deviazione standard σ.



### 1.1 Generazione dataset

E' richiesto un set di immagini con le seguenti specifiche: 

- 8 Immagini di dimensione 512x512

- Formato PNG in scala dei grigi 

- Devono contenere tra i 2 ed i 6 oggetti geometrici 

- Oggetti di colore uniforme su uno sfondo nero

  

<img src="C:\Users\Utente\Desktop\progetto mio\download (3).png" alt="download (3)" style="zoom: 50%;" />



Per i vari test useremo in aggiunta altre due immagini, scelte da internet. Le immagini saranno importate tramite `skimage.io` e affinché siano importate in bianco e nero avranno il flag ”as_gray” impostato a True, saranno inoltre caratterizzate così: 

1. **Immagine Con Testo**: Composta da varie prime pagine di giornale e del testo di grandezza varia. 

2. **Immagine Fotografica**: Che ritrae una persona, con molti dettagli e diverse tonalità di grigio. 

<div style="text-align:center">Le immagini selezionate sono le seguenti:</div>

​                                                     <img src="C:\Users\Utente\Desktop\progetto mio\download (2).png" alt="download (2)" style="zoom: 67%;" />   



### 1.2 Generazione Immagini Corrotte

**Obiettivo**: Degradare le immagini applicando, mediante le funzioni riportate nella cella precedente, l’operatore di blur con parametri 

- σ = 0.5 dimensione 5 × 5
- σ = 1 dimensione 7 × 7
- σ = 1.3 dimensione 9 × 9 

Aggiungendo rumore gaussiano con deviazione standard **(0, 0.05)]**

![set5x5](C:\Users\Utente\Desktop\progetto mio\set5x5.png)

<div style="text-align:center">σ = 0,5 dimensione 5 × 5</div>



![set7x7](C:\Users\Utente\Desktop\progetto mio\set7x7.png)

<div style="text-align:center">σ = 1 dimensione 7 × 7</div>



![set9x9](C:\Users\Utente\Desktop\progetto mio\set9x9.png)

<div style="text-align:center">σ = 1,3 dimensione 9 × 9</div>



### 1.3 Osservazioni

Osserviamo il risultato su un'immagine scelta casualmente del set creato e sulle due immagini aggiuntive: 

La figura che analizziamo variando i valori di sigma è l'immagine numero 8. 



<img src="C:\Users\Utente\Desktop\progetto mio\kernel5img8.png" alt="kernel5img8" style="zoom:50%;" />

<div style="text-align:center">σ = 0,5 dimensione 5 × 5</div>



<img src="C:\Users\Utente\Desktop\progetto mio\kernel7img8.png" alt="kernel7img8" style="zoom:50%;" />

<div style="text-align:center">σ = 1 dimensione 7 × 7</div>



<img src="C:\Users\Utente\Desktop\progetto mio\kernel9img8.png" alt="kernel9img8" style="zoom:50%;" />

<div style="text-align:center">σ = 1,3 dimensione 9 × 9</div>



Ricordiamo che più è alto il valore del PSNR maggiore sarà la vicinanza dell'immagine corrotta rispetto alla versione originale. Le figure di sinistra rappresentano l'immagine originale, invece a destra sono riportate le immagini corrotte con i rispettivi valori di PSNR. Notiamo che all'aumentare delle dimensioni di sigma il valore di PSNR diminuisce che denota un peggioramento della qualità dell'immagine, infatti le immagini subiscono un'appiattimento dell'intensità della scala dei colori e i contorni delle varie figure geometriche perdono di fermezza. Inoltre è curioso notare..

Valutiamo ora l'immagine fotografica: 



<img src="C:\Users\Utente\Desktop\progetto mio\m\kernel5muhammad.png" alt="kernel5muhammad" style="zoom:50%;" />

<div style="text-align:center">σ = 0,5 dimensione 5 × 5</div>



<img src="C:\Users\Utente\Desktop\progetto mio\m\kernel7muhammad.png" alt="kernel7muhammad" style="zoom:50%;" />

<div style="text-align:center">σ = 1 dimensione 7 × 7</div>



<img src="C:\Users\Utente\Desktop\progetto mio\m\kernel9muhammad.png" alt="kernel9muhammad" style="zoom:50%;" />

<div style="text-align:center">σ = 1,3 dimensione 9 × 9</div>



Si nota un'altra volta che all'aumentare delle dimensioni di sigma diminuisce il PSNR e l'immagine perde di incisività, le versioni corrotte benché risultino visivamente peggiori, si riesce ancora a ben distinguere il soggetto in primo piano, anche se sfocato, in tutte le immagini. 

Passando alla valutazione dell'immagine con testo:



<img src="C:\Users\Utente\Desktop\progetto mio\t\kernel5testo.png" alt="kernel5testo" style="zoom:50%;" />

<div style="text-align:center">σ = 0,5 dimensione 5 × 5</div>



<img src="C:\Users\Utente\Desktop\progetto mio\t\kernel7testo.png" alt="kernel7testo" style="zoom:50%;" />

<div style="text-align:center">σ = 1 dimensione 7 × 7</div>



<img src="C:\Users\Utente\Desktop\progetto mio\t\kernel9testo.png" alt="kernel9testo" style="zoom:50%;" />

<div style="text-align:center">σ = 1,3 dimensione 9 × 9</div>



In questa immagine abbiamo una raccolta di prime pagine di giornale che ci permettono di osservare e valutare meglio la differenza tra l'immagine originale e la versione corrotta, per esempio con `σ = 0,5` otteniamo un un immagine con del testo ancora leggibile sebbene meno nitida, la difficoltà inizia ad essere maggior invece con `σ = 1` dove le scritte più piccole diventano quasi illeggibili, con `σ = 1.3` il PSNR diminuisce ancora sebbene non molto rispetto rispetto a sigma uguale a 1, ma in questo caso anche le scritte più grandi, fatta eccezione per i titoli, perdono di chiarezza. 



### 2. Ricostruzione di un immagine rispetto una versione corrotta

**Importante:** per i test useremo d'ora in poi `σ = 1 dimensione 7 × 7` come valore predefinito.

### 2.1 Metodo Gradiente

Descrizione dell'approccio usato....


<img src="C:\Users\Utente\Desktop\progetto mio\img8gradiente.png" alt="img8gradiente" style="zoom: 65%;" />

<img src="C:\Users\Utente\Desktop\progetto mio\imgfotogradiente.png" alt="imgfotogradiente" style="zoom: 67%;" />

<img src="C:\Users\Utente\Desktop\progetto mio\imgtestogradiente.png" alt="imgtestogradiente" style="zoom:67%;" />



### 2.2 Metodo gradiente coniugato

Descrizione metodo....



![img8gradientecon](C:\Users\Utente\Desktop\progetto mio\img8gradientecon.png)

![imgfotogradientecon](C:\Users\Utente\Desktop\progetto mio\imgfotogradientecon.png)

<img src="C:\Users\Utente\Desktop\progetto mio\imgtestogradientecon.png" alt="imgtestogradientecon"  />



Notiamo che tra i due metodi che il primo ci da come risultato delle immagini con un PSNR definitivamente più alto rispetto al secondo, le immagini sono qualitativamente più simili alle immagini originali. 



### 3. Regolarizzazione







### Conclusioni

