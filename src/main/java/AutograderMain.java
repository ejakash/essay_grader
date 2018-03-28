import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class AutograderMain {

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

}
