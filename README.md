# 1. Hauptpunkte dieses Repositorys sind:
- eine **Klasse ``MLPipeline``** (in ml_pipeline.py)
- eine **Web-Applikation ``Machine Learning Application``** bietet eine Oberfläche, welche die Klasse für den Anwender intuitiv nutzbar macht,  
  erreichbar unter:

        https://oookathooomachinelearningapp.streamlit.app/

- sowie eine **Anwendung ``test_app.py``** zum ausführlichen Testen der Klasse incl. Dokumentation der Ergebnisse in einer csv-Datei. Diese App dient gleichzeitig als Anwendungsbeispiel für die Klasse ``MLPipeline``


# 2. MLPipeline
Mithilfe der Klasse ``MLPipeline`` können von ``scikit learn``  bereitgestellte Datensätze geladen werden. Skalierung und eine Dimensionsreduktion der features mittels PCA ermöglicht die Methode ``preprocess_data``. Machine Learning Modelle können mit der Methode ``create_fit_model`` erstellt und trainiert werden. Eine Auswertung erfolgt durch ``evaluate()``, welche anschließend mit ``documentation_to_csv`` in einer Übersicht festgehalten werden kann.
Die Methode ``save_model`` speichert ein Modell an gewünschter Stelle.  

Die wählbaren Datensätze ``Iris``, ``Digits`` sowie ``Breast Cancer``
sind Versuchsdatensätze für Klassifikationsprobleme. Lineare Regression und unsupervised learning Modelle wie Clustering sind für diese Probleme nicht geeignet, wurden aber für einen Vergleich 'passend gemacht' und implementiert.


# 3. Web-Anwendung
Die Web-Anwendung wurde mithilfe von streamlit erstellt und ist in der streamlit-comunity-cloud über die URL 
**https://oookathooomachinelearningapp.streamlit.app/**  
erreichbar.  

Sie bietet eine Oberfläche zur Auswahl verschiedenster Argumente für die Methoden von MLPipeline, welche über den Button ``Run ML Process`` gestartet werden. Auf der Website wird anschließend eine Zusammenfassung des Prozesses und eine Confusion Matrix sichtbar.
Durch Betätigen des Buttons ``Retain Results`` werden die Werte der Zusammenfassung in die darunterliegende Tabelle eingetragen. Diese kann als csv-Datei gespeichert werden. Ein weiterer Button ermöglicht das Ablegen des Models (pickle-Format) im Downloadordner.
Auf der Seite ``dataset information`` sind die Datensätze und zugehörige Informationen einsehbar.


# 4. Installationshinweise
Für die Installation der notwendigen Module steht die Datei ``requirements.txt`` zur verfügung:

        pip install -r requirements.txt


# 5. Start der Applikationen
Nach Download des Repositorys und Installation wie unter Punkt 4 navigiert man via Powershell in den Projektordner und startet die **Test-Anwendung** mit:

        python test_app.py

Möchte man die **Web-Anwendung** nicht online, sondern von dem Projektordner aus starten, lautet der Befehl:

        streamlit run streamlit_app.py
