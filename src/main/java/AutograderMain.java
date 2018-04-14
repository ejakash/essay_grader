import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.IIndexWord;
import edu.mit.jwi.item.POS;
import edu.stanford.nlp.ling.CoreAnnotations;
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
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class AutograderMain {

    private static IDictionary dictionary = null;
    private static HashSet<String> closedWords_en = null;

    static {
        try {
            closedWords_en = new HashSet<>(Files.readAllLines(Paths.get("lib/closed_class.txt")));
        } catch (IOException e) {
            e.printStackTrace();
        }
        URL url = null;
        try {
            url = new URL("file", null, "lib/dict");
        } catch (MalformedURLException e) {
            e.printStackTrace();
        }
        if (url != null) dictionary = new Dictionary(url);
    }

    private static int getLengthScore(Annotation document) {
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

    private static double spellCheck(Annotation document) {
        List<String> tokenLemma = document.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.LemmaAnnotation.class)).collect(Collectors.toList());
        Long correctCount = tokenLemma.stream().filter(AutograderMain::isCorrect).count();//TODO change to wrong count
        return (double) correctCount / tokenLemma.size() * 4;//TODO scale to (low misspells)0,1,2,3,4(high misspells)
    }

    private static boolean isCorrect(String text) {
        if (closedWords_en == null) System.out.println("Unable to read the closed word list");
        else if (closedWords_en.contains(text.toLowerCase())) return true;
        if (dictionary == null) {
            System.out.println("Unable to read the wordnet dictionary");
            //Assume all words to the true if the spelling checker fails to load
            return true;
        }
        try {
            dictionary.open();
        } catch (IOException e) {
            e.printStackTrace();
        }
        POS[] posList = {POS.NOUN, POS.VERB, POS.ADJECTIVE, POS.ADVERB};
        for (POS pos : posList) {
            IIndexWord idxWord = dictionary.getIndexWord(text, pos);
            if (idxWord != null) return true;
        }
        dictionary.close();
        return false;
    }

    private static boolean isSubjListSglr(List<Integer> subjIndBefVerbList, List<String> posList, List<String> wordList, int vi) {
        List<String> nounSgPos = Arrays.asList("NNP", "NN");
        List<String> firstPersSg = Arrays.asList("I", "you", "You");
        List<String> orNor = Arrays.asList("or", "nor");
        if (subjIndBefVerbList.size() == 1){
            if (nounSgPos.contains(posList.get(subjIndBefVerbList.get(0)))){
                return true;
            }else if(posList.get(subjIndBefVerbList.get(0)).equals("PRP")){
                if(!firstPersSg.contains(wordList.get(subjIndBefVerbList.get(0)))){
                    return true;
                }
            }else{
                return false;
            }
        }else if(subjIndBefVerbList.size() > 1){
            int andCount = (int)IntStream.range(subjIndBefVerbList.get(0), vi).filter(k -> wordList.get(k).equals("and")).count();
            int orNorCount = (int)IntStream.range(subjIndBefVerbList.get(0), vi).filter(k -> orNor.contains(wordList.get(k))).count();
            if (andCount >= 1){
                return false;
            }else if (orNorCount >= 1){
                if (nounSgPos.contains(posList.get(subjIndBefVerbList.get(subjIndBefVerbList.size() - 1)))){
                    return true;
                }else if(posList.get(subjIndBefVerbList.get(subjIndBefVerbList.size() - 1)).equals("PRP")){
                    if(!firstPersSg.contains(wordList.get(subjIndBefVerbList.get(subjIndBefVerbList.size() - 1)))){
                        return true;
                    }
                }else{
                    return false;
                }
            }else{
                return false;
            }

        }
        return true;

    }

    private static int getSubjectVerbAgrmntScore(Annotation document) {
        List<String> verbPos = Arrays.asList("VB", "VBP", "VBZ");
        List<String> subjPos = Arrays.asList("NNP", "PRP", "NN", "NNPS", "NNS");
        int mistakeCount = 0;
        for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
            List<String> wordList = sentence.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.TextAnnotation.class)).collect(Collectors.toList());
            List<String> posList = sentence.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.PartOfSpeechAnnotation.class)).collect(Collectors.toList());
            List<Integer> verbIndexList = IntStream.range(0, posList.size()).filter(i -> verbPos.contains(posList.get(i))).boxed().collect(Collectors.toList());
            int viStart = 0;
            Map<Integer, List<Integer>> verbSubjListMap = new HashMap<>();
            for (Integer viEnd : verbIndexList) {
                List<Integer> subjIndBefVerbList = IntStream.range(viStart, viEnd).filter(k -> subjPos.contains(posList.get(k))).boxed().collect(Collectors.toList());
                verbSubjListMap.put(viEnd, subjIndBefVerbList);
                viStart = viEnd;
            }

            for (Map.Entry<Integer, List<Integer>> entry : verbSubjListMap.entrySet()) {
                int vi = entry.getKey();
                List<Integer> subjIndBefVerbList = entry.getValue();
                if (posList.get(vi).equals("VB") || posList.get(vi).equals("VBP")) {
                    if(isSubjListSglr(subjIndBefVerbList, posList, wordList, vi)){
                        mistakeCount++;
                    }
                } else if (posList.get(vi).equals("VBZ")) {
                    if(!isSubjListSglr(subjIndBefVerbList, posList, wordList, vi)){
                        mistakeCount++;
                    }
                }
            }
        }
        //TODO generate score from mistake count
        return 0;
    }

    public static void main(String[] args) {
        try {
            Reader reader = Files.newBufferedReader(Paths.get("input/training/index.csv"));
            CSVParser csvParser = new CSVParserBuilder().withSeparator(';').build();
            CSVReader csvReader = new CSVReaderBuilder(reader).withCSVParser(csvParser).withSkipLines(1).build();
            String[] nextRecord;

            Properties props = new Properties();
            props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
            StanfordCoreNLP pipeline = new StanfordCoreNLP(props);


            while ((nextRecord = csvReader.readNext()) != null) {
                BufferedReader essayReader = Files.newBufferedReader(Paths.get("input/training/essays/" + nextRecord[0]));
                StringBuilder essay = new StringBuilder();
                String line;
                while ((line = essayReader.readLine()) != null) {
                    essay.append(line).append("\n");
                }

                Annotation document = new Annotation(essay.toString());
                pipeline.annotate(document);
                int lengthScore = getLengthScore(document);
                double spellScore = spellCheck(document);
                int subjVerbAgrmntScore = getSubjectVerbAgrmntScore(document);
                System.out.println(lengthScore + "\t" + spellScore + "\t" + nextRecord[2]);

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
