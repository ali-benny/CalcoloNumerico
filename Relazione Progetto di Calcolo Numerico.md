Relazione Progetto di Calcolo Numerico

Benatti Alice, Manuelli Matteo, Qayyum Shahbaz Ali 

*Alma Mater Studiorum - Università di Bologna* 

### 1.Introduzione

Per svolgere il progetto si farà uso dei moduli `numpy , skimage e matplotlib` utilizzando il linguaggio Python. Il progetto ha come scopo quello di comprendere e mettere in atto metodi per ricostruire immagini blurrate e svolgere il lavoro opposto, quindi generare immagini corrotte (dal rumore) a partire da un immagine originale. 

Il problema che ci è stato presentato riguarda la ricostruzione di immagini corrotte attraverso il blur Gaussiano. Verrà analizzata inizialmente l’immagine `data.camera()` importata da `skimage` , successivamente verranno analizzate un set di 8 immagini con oggetti geometrici di colore uniforme su sfondo nero, realizzate da noi. Il problema di deblur consiste nella ricostruzione di un immagine a partire da un dato acquisito mediante il seguente modello:

<div style="text-align:center">b = Ax + η</div>

Dove b rappresenta l’immagine corrotta, x l’immagine originale che vogliamo ricostruire, A l’operatore che applica il blur Gaussiano ed η il rumore additivo con distribuzione Gaussiana di media ⊬ e deviazione standard σ.

Affinché risultino chiari i valori a cui andremo a riferirci nella relazione, bisogna tenere ben presente il significato di questi due parametri. 

**PSNR** (Peak Signal to Noise Ratio): Misura la qualità di un immagine ricostruita rispetto all'immagine originale, la formula per calcolarlo è la seguente:  $PSNR = log_{10}(\frac{max\;x^\ast}{\sqrt{MSE}})$

**MSE** (Mean Squared Error):  Con la sigla ci riferiamo all'errore quadratico medio ed è così ottenuto: $MSE = \sqrt[2]{\frac{\sum_{i=1}^n\sum_{j=1}(x^{\ast}_{ij}-x_{ij})}{nm}}$

I due valori sono inversamente proposizionali, quindi più è alto il PSNR e basso l'MSE, più l'immagine sarà simile all'immagine originale. il PSNR dipende dall'MSE. 

**Deviazione standard**: E' un indice che ci permette di capire in maniera riassuntiva le differenze dei valori per ogni osservazioni rispetto alla media delle variabili. 



### 1.1 Generazione dataset

E' richiesto un set di immagini con le seguenti specifiche: 

- 8 Immagini di dimensione 512x512

- Formato PNG in scala dei grigi 

- Devono contenere tra i 2 ed i 6 oggetti geometrici 

- Oggetti di colore uniforme su uno sfondo nero

  

<img src="C:\Users\Utente\Desktop\progetto mio\download (3).png" alt="download (3)" style="zoom: 50%;" />



Per i vari test useremo in aggiunta altre due immagini, scelte da internet. Le immagini saranno importate tramite `skimage.io` e affinché siano importate in bianco e nero avranno il flag `”as_gray”` impostato a True, saranno inoltre caratterizzate così: 

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

![datasetcorrotto5x5](C:\Users\Utente\Desktop\Progetto aggiornato\datasetcorrotto\datasetcorrotto5x5.png)

<div style="text-align:center">σ = 0,5 dimensione 5 × 5</div>

|     Valore     | Risultato |
| :------------: | :-------: |
|   Media PSNR   |  25.3137  |
|   Media MSE    | 0.002943  |
| Dev. Std. MSE  | 0.000106  |
| Dev. Std. PSNR | 0.156402  |



![datasetcorrotto7x7](C:\Users\Utente\Desktop\Progetto aggiornato\datasetcorrotto\datasetcorrotto7x7.png)

<div style="text-align:center">σ = 1 dimensione 7 × 7</div>

|     Valore     | Risultato |
| :------------: | :-------: |
|   Media PSNR   |  24.7938  |
|   Media MSE    | 0.003321  |
| Dev. Std. MSE  | 0.000194  |
| Dev. Std. PSNR | 0.252202  |



![datasetcorrotto9x9](C:\Users\Utente\Desktop\Progetto aggiornato\datasetcorrotto\datasetcorrotto9x9.png)

<div style="text-align:center">σ = 1,3 dimensione 9 × 9</div>

|     Valore     | Risultato |
| :------------: | :-------: |
|   Media PSNR   |  24.6292  |
|   Media MSE    | 0.003452  |
| Dev. Std. MSE  | 0.000235  |
| Dev. Std. PSNR | 0.293261  |



### 1.3 Osservazioni

Osserviamo il risultato su un'immagine scelta casualmente del set creato e sulle due immagini aggiuntive: 

La figura che analizziamo variando i valori di sigma è l'immagine numero 8. 



![img8corrotta5x5](C:\Users\Utente\Desktop\Progetto aggiornato\img8corrotto\img8corrotta5x5.png)

<div style="text-align:center">σ = 0,5 dimensione 5 × 5</div>



![img8corrotta7x7](C:\Users\Utente\Desktop\Progetto aggiornato\img8corrotto\img8corrotta7x7.png)

<div style="text-align:center">σ = 1 dimensione 7 × 7</div>



![img8corrotta9x9](C:\Users\Utente\Desktop\Progetto aggiornato\img8corrotto\img8corrotta9x9.png)

<div style="text-align:center">σ = 1,3 dimensione 9 × 9</div>



Ricordiamo che più è alto il valore del PSNR maggiore sarà la vicinanza dell'immagine corrotta rispetto alla versione originale. Le figure di sinistra rappresentano l'immagine originale, invece a destra sono riportate le immagini corrotte con i rispettivi valori di PSNR. Notiamo che all'aumentare delle dimensioni di sigma il valore di PSNR diminuisce che denota un peggioramento della qualità dell'immagine, infatti le immagini subiscono un'appiattimento dell'intensità della scala dei colori e i contorni delle varie figure geometriche perdono di fermezza. Inoltre è curioso notare..

Valutiamo ora l'immagine fotografica: 



![muhammedcorrotto5x5](C:\Users\Utente\Desktop\Progetto aggiornato\muhammedcorrotto\muhammedcorrotto5x5.png)

<div style="text-align:center">σ = 0,5 dimensione 5 × 5</div>



![muhammedcorrotto7x7](C:\Users\Utente\Desktop\Progetto aggiornato\muhammedcorrotto\muhammedcorrotto7x7.png)

<div style="text-align:center">σ = 1 dimensione 7 × 7</div>



![muhammedcorrotto9x9](C:\Users\Utente\Desktop\Progetto aggiornato\muhammedcorrotto\muhammedcorrotto9x9.png)

<div style="text-align:center">σ = 1,3 dimensione 9 × 9</div>



Si nota un'altra volta che all'aumentare delle dimensioni di sigma diminuisce il PSNR e l'immagine perde di incisività, le versioni corrotte benché risultino visivamente peggiori, si riesce ancora a ben distinguere il soggetto in primo piano, anche se sfocato, in tutte le immagini. 

Passando alla valutazione dell'immagine con testo:



![testocorrotto5x5](C:\Users\Utente\Desktop\Progetto aggiornato\testocorrotto\testocorrotto5x5.png)

<div style="text-align:center">σ = 0,5 dimensione 5 × 5</div>



![testocorrotto7x7](C:\Users\Utente\Desktop\Progetto aggiornato\testocorrotto\testocorrotto7x7.png)

<div style="text-align:center">σ = 1 dimensione 7 × 7</div>



![testocorrotto9x9](C:\Users\Utente\Desktop\Progetto aggiornato\testocorrotto\testocorrotto9x9.png)

<div style="text-align:center">σ = 1,3 dimensione 9 × 9</div>



In questa immagine abbiamo una raccolta di prime pagine di giornale che ci permettono di osservare e valutare meglio la differenza tra l'immagine originale e la versione corrotta, per esempio con `σ = 0,5` otteniamo un un immagine con del testo ancora leggibile sebbene meno nitida, la difficoltà inizia ad essere maggior invece con `σ = 1` dove le scritte più piccole diventano quasi illeggibili, con `σ = 1.3` il PSNR diminuisce ancora sebbene non molto rispetto rispetto a sigma uguale a 1, ma in questo caso anche le scritte più grandi, fatta eccezione per i titoli, perdono di chiarezza. 



### 2. Ricostruzione di un immagine rispetto una versione corrotta

Una possibile ricostruzione dell'immagine originale $x$ partendo dall'immagine corrotta $b$ è la soluzione naive data dal minimo del seguente problema di ottimizzazione:                                                      

​                                                     $x^* = \arg\min_x \frac{1}{2} ||Ax - b||_2^2$

**Importante:** per i test useremo d'ora in poi `σ = 1 dimensione 7 × 7` con deviazione standard **(0, 0.05)]**  come valori predefiniti.



![image-20220113193214230](C:\Users\Utente\AppData\Roaming\Typora\typora-user-images\image-20220113193214230.png)

Abbiamo mostrato questi due grafici sovrastanti perché si evince che all'aumentare del numero delle iterazioni l'immagine ricavata si allontana dalla sua versione originale poiché è presente una deviazione. 



### 2.1 Metodo Gradiente Coniugato

Il metodo del gradiente coniugato è un algoritmo per la risoluzione numerica di un sistema lineare la cui matrice sia simmetrica e definita positiva e consente di risolvere il sistema in un numero di iterazioni che e' al massimo $n$.

La funzione $f$ da minimizzare è data dalla formula $f(x) = \frac{1}{2} ||Ax - b||_2^2 $, il cui gradiente $\nabla f$ è dato da $\nabla f(x) = A^TAx - A^Tb$.

Utilizzando il metodo del gradiente coniugato implementato dalla funzione `minimize`abbiamo calcolato la soluzione naive.



![datasetconiugato](C:\Users\Utente\Desktop\Progetto aggiornato\datasetconiugato.png)

<img src="C:\Users\Utente\Desktop\Progetto aggiornato\immaginigradienteconiugato.png" alt="immaginigradienteconiugato" style="zoom:50%;" />





### 2.2 Metodo Del Gradiente

Il metodo del gradiente è un algoritmo che calcola il vettore di minimo globale, ovvero: Un vettore $x^{\ast}$ è un punto di minimo globale di $f(x)$ se $f(x^{\ast}) \leq f(x) \forall x \in R^n$.

 Analogamente, un vettore $x^{\ast}$ è un punto di minimo globale in senso stretto di $f(x) se
f(x^{\ast}) < f(x) \forall x \in R+n \and x \neq x^{\ast}$.

![datasetgradiente](C:\Users\Utente\Desktop\Progetto aggiornato\datasetgradiente.png)

<img src="C:\Users\Utente\Desktop\Progetto aggiornato\immaginigradiente.png" alt="immaginigradiente" style="zoom:50%;" />

<img src="C:\Users\Utente\Desktop\Progetto aggiornato\immagine2gradienteù.png" alt="immagine2gradienteù" style="zoom:50%;" />

Notiamo che tra i due metodi che il primo ci da come risultato delle immagini con un PSNR definitivamente più alto rispetto al secondo, le immagini sono qualitativamente più simili alle immagini originali. 



### 3. Regolarizzazione

I metodi di regolarizzazione rinunciano a trovare la soluzione esatta del problema del precedente problema di ottimizzazione, ma invece calcolano la soluzione di un problema leggermente diverso ma meglio condizionato. Quest’ultimo viene chiamato problema regolarizzato. 

### 3.1 Metodo di Regolarizzazione di Tikhonov

Per ridurre gli effetti del rumore nella ricostruzione `e necessario introdurre un termine di regolarizzazione di Tikhonov. Si considera quindi il seguente problema di ottimizzazione.

Si deve risolvere  $ Ax_\epsilon = b_\epsilon$ con $b_\epsilon = b+\epsilon$ , invece di risolvere direttamente il sistema lineare (se quadrato) o di minimizzare la
norma 2 del residuo (se il sistema è rettangolare)  $||Ax_\epsilon -b_\epsilon||_2^2$, si aggiunge un vincolo di regolarità alla soluzione e si minimizza, ad esempio $||Ax_\epsilon-b_\epsilon||_2^2+\gamma_\epsilon||x_\epsilon||_2^2$ che rappresenta la **forma standard** della regolarizzazione di Tikhonov.

Analizziamo i grafici ottenuti cercando di ridurre il rumore nella ricostruzione delle immagini del dataset.   

  <img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputPSNR-img1.png" alt="outputPSNR-img1" style="zoom: 70%;" /><img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputMSE-img1.png" alt="outputMSE-img1" style="zoom: 70%;" />

  <img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputPSNR-img2.png" alt="outputPSNR-img2" style="zoom: 70%;" /><img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputMSE-img2.png" alt="outputMSE-img2" style="zoom:70%;" />

 <img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputPSNR-img3.png" alt="outputPSNR-img3" style="zoom:70%;" /><img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputMSE-img3.png" alt="outputMSE-img3" style="zoom:70%;" />

  <img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputPSNR-img4.png" alt="outputPSNR-img4" style="zoom:70%;" /><img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputMSE-img4.png" alt="outputMSE-img4" style="zoom:70%;" />

 <img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputPSNR-img5.png" alt="outputPSNR-img5" style="zoom:70%;" /><img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputMSE-img5.png" alt="outputMSE-img5" style="zoom:70%;" />

<img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputPSNR-img6.png" alt="outputPSNR-img6" style="zoom: 70%;" /><img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputMSE-img6.png" alt="outputMSE-img6" style="zoom: 70%;" />

 <img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputPSNR-img7.png" alt="outputPSNR-img7" style="zoom:70%;" /><img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputMSE-img7.png" alt="outputMSE-img7" style="zoom:70%;" />

 <img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputPSNR-img8.png" alt="outputPSNR-img8" style="zoom:70%;" /><img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\outputMSE-img8.png" alt="outputMSE-img8" style="zoom:70%;" />



<div style="text-align:center">La colonna a sinistra riporta i grafici riguardanti il PSNR, mentre a destra troviamo i grafici con gli MSE al variare di lambda.</div>



Per quanto riguarda le immagini fotografiche, abbiamo riscontrato un curioso errore. 



<img src="C:\Users\Utente\Desktop\Nuova cartella (2)\CalcoloNumerico-main\output\image_2022-01-13_19-48-50.png" alt="image_2022-01-13_19-48-50" style="zoom: 60%;" />

Possiamo affermare che l'incremento è talmente impercettibile che non viene colto dal calcolatore poiché inferiore alla sua precisione macchina, con la regolarizzazione si ottengono comunque risultati migliori. 



### 4. Variazione Totale. 

Tramite un algoritmo possiamo recuperare immagini sfocate basandoci sulla Variazione totale partendo da una Blurring Point-Spread function di un'immagine. 

La variazione totale è definita dalla seguente formula:                                                                                

​                                            $TV(u) = \sum_i^n{\sum_j^m{\sqrt{||\nabla u(i, j)||_2^2 + \epsilon^2}}}$

Per calcolare il gradiente dell'immagine $\nabla u$ usiamo la funzione np.gradient che approssima la derivata per ogni pixel calcolando la differenza tra pixel adiacenti. I risultati sono due immagini della stessa dimensione dell'immagine in input, una che rappresenta il valore della derivata orizzontale dx e l'altra della derivata verticale dy. Il gradiente dell'immagine nel punto $(i, j)$ è quindi un vettore di due componenti, uno orizzontale contenuto in dx e uno verticale in dy.

​                                          $\begin{align*}
  x^* = \arg\min_x \frac{1}{2} ||Ax - b||_2^2 + \lambda TV(u)
\end{align*}$
il cui gradiente $\nabla f$ è dato da

​                      immagine A^Tb)  + \lambda \nabla TV(x)
\end{align*}$



![graficoPugile](C:\Users\Utente\Desktop\Progetto aggiornato\graficoPugile.png)

<img src="C:\Users\Utente\Desktop\Progetto aggiornato\pugile.png" alt="pugile" style="zoom: 50%;" />

<div style="text-align:center">Figura.1: Immagine Originale, Figura.2: Immagine Corrotta, Figura.3: Immagine Ricostruita </div>



![graficoGiornale](C:\Users\Utente\Desktop\Progetto aggiornato\graficoGiornale.png)

<img src="C:\Users\Utente\Desktop\Progetto aggiornato\giornale.png" alt="giornale" style="zoom:50%;" />

<div style="text-align:center">Figura.1: Immagine Originale, Figura.2: Immagine Corrotta, Figura.3: Immagine Ricostruita </div>





### Conclusioni

.

