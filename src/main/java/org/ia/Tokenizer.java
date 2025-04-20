package org.ia;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * La classe {@code Tokenizer} viene usata per convertire un prompt dato in input in valori numerici comprensibili dal modello, e viceversa.
 */
public class Tokenizer {
    private Map<Integer, String> vocab;

    public Tokenizer(Map<Integer, String> vocab){
        this.vocab = vocab;
    }

    /**
     * Converte un prompt in input, suddividendolo in token. Ogni parola viene processata e i token vengono aggiunti in una lista.
     * Viene aggiunto un token di separazione (13) per separare i token.
     * @param prompt Il prompt in input, che e' una stringa
     * @return Una lista di token corrispondenti al prompt in input
     */
    public List<Integer> promptTokens(String prompt){
        List<Integer> tokenList = new ArrayList<>();
        String[] words = prompt.split(" ");

        for (String word: words){
            for (int token : processWord(word)){
                tokenList.add(token);
            }
            tokenList.add(13);
        }

        if (!tokenList.isEmpty() && tokenList.getLast() == 13){
            tokenList.removeLast();
        }

        return tokenList;
    }

    /**
     * Processa una parola singola, cercando i token corrispondenti nel vocabolario. La parola viene suddivisa in
     * sottostringhe fino a trovare una corrispondenza nel vocabolario. Se non viene trovata la parola viene suddivisa carattere
     * per carattere.
     * @param word La parola da analizzare
     * @return Un array di interi che rappresentano i token corrispondenti alla parola.
     */
    private int[] processWord(String word) {
        List<Integer> tokens = new ArrayList<>();
        int start = 0;

        while (start < word.length()) {
            int end = word.length();
            boolean matched = false;

            while (end > start) {
                String sub = word.substring(start, end);
                Integer token = findInVocab(sub);
                if (token != null) {
                    tokens.add(token);
                    start = end;
                    matched = true;
                    break;
                }
                end--;
            }

            if (!matched) {
                start++;
            }
        }

        int[] result = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            result[i] = tokens.get(i);
        }
        return result;
    }

    /**
     * Cerca una parola nel vocabolario e restituisce l'indice del token corrispondente
     * @param word La parola da cercare nel vocabolario
     * @return L'indice del token corrispondente se trovato, altrimenti {@code null}
     */
    private Integer findInVocab(String word){
        for (Map.Entry<Integer, String> entry : vocab.entrySet()){
            if (entry.getValue().equals(word)){
                return entry.getKey();
            }
        }
        return null;
    }

    /**
     * Converte una lista di token in una stringa di testo. Ogni token viene convertito tramite il metodo {@code decodeToken}
     * @param tokens La lista di token da convertire
     * @return Una stringa che rappresenta il testo convertito
     */
    public String decodeTokens(List<Integer> tokens){
        StringBuilder sb = new StringBuilder();
        for (int token : tokens){
            sb.append(decodeToken(token));
        }
        return sb.toString();
    }

    /**
     * Converte un token in una stringa, restituendola.
     * @param tokenId L'ID del token da convertire
     * @return La stringa che rappresenta il token
     */
    private String decodeToken(int tokenId){
        return vocab.getOrDefault(tokenId, "<UNK>");
    }
}
