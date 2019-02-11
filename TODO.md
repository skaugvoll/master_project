

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