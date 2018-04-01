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
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.trees.TypedDependency;
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
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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

       public static int getLengthScore(Annotation document) {
        int sentenceCount = 0;
        String[] sepArr = {"CC", "IN", ",", "WRB", "WDT", "WP", "WP$"};
        for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
            SemanticGraph dependencyParse =
                    sentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);

            int sentenceSplitCount = 0;
            List<Integer> subjIndexList = new ArrayList<>();
            for (TypedDependency t : dependencyParse.typedDependencies()) {
                if (t.reln().toString().contains("subj")) {
                    String s = t.dep().originalText();
                    if (Character.isUpperCase(s.charAt(0))) {
                        subjIndexList.add(t.dep().index());
                    }
                }
            }
            List<String> posList = sentence.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.PartOfSpeechAnnotation.class)).collect(Collectors.toList());
            Collections.sort(subjIndexList);
            if (subjIndexList.size() > 1) {
                for (int i = 1; i < subjIndexList.size(); i++) {
                    int sepCount = (int) IntStream.range(subjIndexList.get(i - 1), subjIndexList.get(i))
                            .filter(k -> (Arrays.asList(sepArr).contains(posList.get(k - 1)))).count();
                    if (sepCount == 0) {
                        sentenceSplitCount++;
                    }
                }
            }
            if (sentenceSplitCount > 1) {
                sentenceCount += sentenceSplitCount;
            } else {
                sentenceCount++;
            }
        }
        int lengthScore = 1;
        if (sentenceCount >= 10) {
            lengthScore++;
        }
        if (sentenceCount >= 13) {
            lengthScore++;
        }
        if (sentenceCount >= 16) {
            lengthScore++;
        }
        if (sentenceCount >= 20) {
            lengthScore++;
        }
        return lengthScore;
    }

    public static void main(String[] args) {
        try {
            Reader reader = Files.newBufferedReader(Paths.get("essays_dataset/index.csv"));
            CSVParser csvParser = new CSVParserBuilder().withSeparator(';').build();
            CSVReader csvReader = new CSVReaderBuilder(reader).withCSVParser(csvParser).withSkipLines(1).build();
            String[] nextRecord;

            Properties props = new Properties();
            props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
            StanfordCoreNLP pipeline = new StanfordCoreNLP(props);


            while ((nextRecord = csvReader.readNext()) != null) {
                BufferedReader essayReader = Files.newBufferedReader(Paths.get("essays_dataset/essays/" + nextRecord[0]));
                StringBuilder essay = new StringBuilder();
                String line;
                while ((line = essayReader.readLine()) != null) {
                    essay.append(line).append("\n");
                }

                Annotation document = new Annotation(essay.toString());
                pipeline.annotate(document);
                int lengthScore = getLengthScore(document);
                //System.out.println(lengthScore + "\t" + nextRecord[2]);

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
