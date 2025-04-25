package org.ia;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * La classe {@code Methods} contiene metodi usati per caricare il vocabolario,
 * applicare la funzione di softmax e calcolare l'argmax di un array di valori.
 */
public class Methods {

    private Map<Integer, String> vocabMap = new HashMap<>();

    /**
     * Carica il vocabolario da un file JSON e lo converte in una mappa.</br>
     * La mappa associa ogni indice di token alla stringa corrispondente
     * @param resourcePath Il percorso del file JSON contentente il vocabolario
     * @return Una mappa di token, in cui la chiave e' l'indice e il valore e' la stringa del token
     */
    public Map<Integer, String> loadVocabulary(String resourcePath) {
        try (InputStream inputStream = getClass().getClassLoader().getResourceAsStream(resourcePath)) {
            if (inputStream == null) {
                throw new FileNotFoundException("Il file " + resourcePath + " non è stato trovato nel classpath.");
            }

            ObjectMapper objectMapper = new ObjectMapper();
            Map<String, Integer> rawVocab = objectMapper.readValue(inputStream, Map.class);

            for (Map.Entry<String, Integer> entry : rawVocab.entrySet()) {
                vocabMap.put(entry.getValue(), entry.getKey());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return vocabMap;
    }

    /**
     * Applica la funzione di softmax a un array di logits e restituisce un array di probabilita'.</br>
     * La funzione softmax normalizza i valori in modo che la somma delle probabilita' sia 1.
     * @param logits Un array di valori logits, uno per ciascun token
     * @return Un array di probabilita' corrispondente ai logits di ingresso
     */
    public float[] applySoftmax(float[] logits){
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float logit : logits){
            maxLogit = Math.max(maxLogit, logit);
        }
        float sumExp = 0;
        for (int i = 0; i < logits.length; i++){
            logits[i] = (float) Math.exp(logits[i] - maxLogit);
            sumExp += logits[i];
        }

        for (int i = 0; i < logits.length; i++){
            logits[i] /= sumExp;
        }
        return logits;
    }

    /**
     * Calcola l'indice dell'elemento massimo in un array.</br>
     * Questo metodo viene usato per determinare il token successivo da generare in base ai logits
     * @param array Un array di valori numerici
     * @return L'indice dell'elemento con il valore massimo
     */
    public int argmax(float[] array){
        int index = 0;
        for (int i = 0; i < array.length; i++){
            if (array[i] > array[index]){
                index = i;
            }
        }
        return index;
    }
}
