package org.ia;

import ai.onnxruntime.*;

import java.io.FileNotFoundException;
import java.nio.file.*;
import java.util.*;

/**
 * La classe {@code Model} gestisce il caricamento e l'avvio di un modello in linguaggio ONNX.<br/>
 * Permette di inserire un prompt testuale, convertirlo in token leggibili per il modello, inviarlo al modello e generare una risposta.
 */
public class Model {

    private String vocabPath = "./TinyLlama/vocab.json";
    private String modelRelativePath = "src/main/resources/TinyLlama/onnx/decoder_model.onnx";
    Path modelAbsolutePath = Paths.get(modelRelativePath).toAbsolutePath();
    private String[] states = {"Generazione risposta .", "Generazione risposta ..", "Generazione risposta ..."};
    private Methods methods = new Methods();
    private Map<Integer, String> vocab = methods.loadVocabulary(vocabPath);
    private Tokenizer tokenizer = new Tokenizer(vocab);
    private String userPrompt;
    private int maxToken;

    /**
     * Avvia il modello ONNX per generare una risposta a partire da un prompt inserito dall'utente. Il metodo:</br>
     * <ul>
     *      <li>Legge in input il prompt e il numero massimo di token da generare</li>
     *      <li>Converte il prompt in tokens usando il metodo {@code promptTokens} </li>
     *      <li>Esegue un ciclo per generare una risposta tramite il modello ONNX</li>
     *      <li>Converte i tokens generati in un prompt leggibile tramite il metodo {@code decodeTokens}</li>
     * </ul>
     * @throws OrtException se si verifica un errore durante l'esecuzione del modello
     */
    public void startModel() throws OrtException, FileNotFoundException {
        Scanner input = new Scanner(System.in);
        System.out.print("Inserire il prompt (solo inglese): ");
        userPrompt = input.nextLine();
        System.out.print("Token utlizzabili: ");
        try {
            maxToken = input.nextInt();
        } catch (InputMismatchException e) {
            System.out.print("Errore!");
            return;
        }

        List<Integer> promptToken = tokenizer.promptTokens(userPrompt);
        List<Integer> answerTokens = new ArrayList<>();

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();

        OrtSession session = env.createSession(String.valueOf(modelAbsolutePath), options);

        for (int step = 0; step < maxToken; step++) {
            List<Integer> inputTokens = new ArrayList<>(promptToken);
            inputTokens.addAll(answerTokens);

            long[][] inputArray = new long[1][inputTokens.size()];
            for (int i = 0; i < inputTokens.size(); i++) {
                inputArray[0][i] = inputTokens.get(i);
            }

            long[][] attentionMaskArray = new long[1][inputArray[0].length];
            for (int i = 0; i < inputArray[0].length; i++) {
                attentionMaskArray[0][i] = 1L;
            }

            OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray);
            OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMaskArray);

            Map<String, OnnxTensor> feeds = new HashMap<>();
            feeds.put("input_ids", inputTensor);
            feeds.put("attention_mask", attentionMaskTensor);
            OrtSession.Result result = session.run(feeds);

            float[][][] outputData = (float[][][]) result.get(0).getValue();
            float[] logits = outputData[0][outputData[0].length - 1];

            int nextToken = methods.argmax(methods.applySoftmax(logits));
            answerTokens.add(nextToken);

            int i = (step + 1) % states.length;
            System.out.print("\r" + states[i]);

            if (nextToken == 50256) break;
        }
        String generatedText = tokenizer.decodeTokens(answerTokens);
        generatedText = generatedText.replace("0x0A", " ").replace("Ċ", " ").replace("Ġ", " ").replace("< >", " ");
        System.out.println("\nDomanda: " + userPrompt + "\nRisposta: " + generatedText);
    }

}

class Main{
    public static void main(String[] args){
        Model model = new Model();
        try {
            model.startModel();
        } catch (OrtException e) {
            System.out.println("Errore nell'avviare il modello. Riavviare!");
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            System.out.println("Errore nel trovare il modello. Verificare i file e riprovare!");
            e.printStackTrace();
        }
    }
}