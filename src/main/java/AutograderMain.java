import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.POS;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.stream.Collectors;

public class AutograderMain {

    private static IDictionary dictionary = null;
    private static HashSet<String> closedWords_en = null;
    static {
        try{closedWords_en = new HashSet<>(Files.readAllLines(Paths.get("lib/closed_class.txt")));}
        catch (IOException e) {e.printStackTrace();}
        URL url = null;
        try{ url = new URL("file", null, "lib/dict"); }
        catch(MalformedURLException e){ e.printStackTrace(); }
        if(url != null)dictionary = new Dictionary(url);
    }

    public static List<String> tokenize(String text) {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // create an empty Annotation just with the given text
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        pipeline.annotate(document);
        List<CoreLabel> tokens = document.get(CoreAnnotations.TokensAnnotation.class);

        List<String> result = new ArrayList<>();
        for (CoreLabel token : tokens) {
            // this is the text of the token
            String word = token.get(CoreAnnotations.TextAnnotation.class);
            result.add(word);
        }

        return result;
    }

    public static List<String> sentenceSplit(String text) {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // create an empty Annotation just with the given text
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        pipeline.annotate(document);
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);

        List<String> result = new ArrayList<>();
        for (CoreMap sentence : sentences) {
            String sentenceString = sentence.get(CoreAnnotations.TextAnnotation.class);
            result.add(sentenceString);

            // see tokenize(String) method
            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
            for (CoreLabel token : tokens) {
                String word = token.get(CoreAnnotations.TextAnnotation.class);
            }
        }

        return result;
    }

    public static List<String> posTagging(String text) {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // create an empty Annotation just with the given text
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        pipeline.annotate(document);
        List<CoreLabel> tokens = document.get(CoreAnnotations.TokensAnnotation.class);

        List<String> result = new ArrayList<>();
        for (CoreLabel token : tokens) {
            // this is the text of the token
            String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
            result.add(token + "/" + pos);
        }

        return result;
    }

    public static void main(String[] args) {
        try {
            Reader reader = Files.newBufferedReader(Paths.get("essays_dataset/index.csv"));
            CSVParser csvParser = new CSVParserBuilder().withSeparator(';').build();
            CSVReader csvReader = new CSVReaderBuilder(reader).withCSVParser(csvParser).withSkipLines(1).build();
            // Reading Records One by One in a String array
            String[] nextRecord;
            int count = 0;
            while ((nextRecord = csvReader.readNext()) != null) {
                System.out.println("Index : " + ++count);
                System.out.println("Filename : " + nextRecord[0]);
                System.out.println("Prompt : " + nextRecord[1]);
                System.out.println("Grade : " + nextRecord[2]);

                System.out.println("==========================");
                BufferedReader essayReader = Files.newBufferedReader(Paths.get("essays_dataset/essays/" + nextRecord[0]));
                StringBuilder essay = new StringBuilder();
                String line;
                while((line = essayReader.readLine()) != null){
                    essay.append(line).append("\n");
                }
//                System.out.println(sentenceSplit(essay.toString()));
//                System.out.println(posTagging(essay.toString()));

            }


        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static double spellCheck(String essay) {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // create an empty Annotation just with the given text
        Annotation document = new Annotation(essay);

        // run all Annotators on this text
        pipeline.annotate(document);
        List<String> tokenLemma = document.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.LemmaAnnotation.class)).collect(Collectors.toList());
        Long correctCount = tokenLemma.stream().filter(AutograderMain::isCorrect).count();
        return (double)correctCount/tokenLemma.size() * 4;
    }

    private static boolean isCorrect(String text)  {
        if(closedWords_en == null) System.out.println("Unable to read the closed word list");
        else if(closedWords_en.contains(text.toLowerCase())) return true;
        if(dictionary == null) {
            System.out.println("Unable to read the wordnet dictionary");
            //Assume all words to the true if the spelling checker fails to load
            return true;
        }
        try {dictionary.open();}
        catch (IOException e) {e.printStackTrace();}
        POS[] posList = {POS.NOUN, POS.VERB, POS.ADJECTIVE, POS.ADVERB};
        for(POS pos : posList) {
            IIndexWord idxWord = dictionary.getIndexWord(text, pos);
            if(idxWord != null) return true;
        }
        dictionary.close();
        return false;
    }

}
