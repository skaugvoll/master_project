

Create a new pipeline
- create a new pipeline directory
- nytt script eller dataImport class
  - splitte hannes get_csv_file funksjon inn i 3 metoder.
  - en metode for å sjekke fileformate på input (ferdigsynka csv vs cwa (7z zip))
     - en metode for hvert format
  - Klasse objektet ogsaa har en metode for aa generete Pandas dataframe av inputfilen(e).

> NB han lager temprary files hvis han unzipper og synker. Veldig lurt! Behold det.

- Lag ny enkel model for aa teste paa daten (byttes ut med meta-classifier etter hvert som vi ser at data-import og gen funker)

Hvordan organisere pipeline koden, module, package,osv ? for aa faa robust import statements osv...


- Kan ikke kjøre samme input flere ganger, med mindre man cleaner up /data/temp/<subjectFolder>
  - Lage en funksjon som bare while not
  tar filnavne og legger på en _ også teller oppover til den får ett tall som gjør den unik.
  lager temp mappen med det navnet.


- Importere flere test dataset og lage labels på de. type 4000181, etc

øystein used subjects 006 trough 022 assuming both sensors are on!
use 006 trough 007 respective thigh and back files for traiing the indivudal LSTMS

husk å skriv antagelser om at øystein brukte begge sensorer og de gikk med gopro kamera for å fange aktivitet,
derfor kan vi anta at thigh filen har bare valid sensor data og back file har bare valid data, altså at sensoren var på hele tiden
og derfor kan vi trene both, thigh og back LSTM pa de dataene for å reprodusere hans resultater.

Så det vi må er å gjøre RFC bedre til å se om det ikke er sensor på, og da bare drite i å klassifisere de windows med ikke sensor på.

Jeg tror at vi må endre hvordan RFC klassifiserer windows til å klassifisere alle timestamps i window til en LSTM, også ta most frequent LSTM som faktisk LSTM å bruke eller ikke klassifisere at all.
