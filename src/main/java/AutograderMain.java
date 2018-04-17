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

    private static <N extends Number> int findIntervalIndex(Number searchValue, List<N> intervals) {
        for (int i = 0; i < intervals.size() - 1; i++) {
            if (intervals.get(i).doubleValue() <= searchValue.doubleValue() && intervals.get(i + 1).doubleValue() > searchValue.doubleValue())
                return i;
        }
        return intervals.size() - 1;
    }

    /**
     * LENGTH
     **/
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
                    if (Character.isUpperCase(s.charAt(0))) { //TODO fail:John and <Jane/his dog> went for a walk Jack ate food.
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

        List<Integer> values = Arrays.asList(0, 10, 13, 16, 20);
        return findIntervalIndex(sentenceCount, values) + 1;

    }

    /**
     * SPELLING
     **/
    private static int spellCheck(Annotation document) {
        List<String> tokenLemma = document.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.LemmaAnnotation.class)).collect(Collectors.toList());
        Long correctCount = tokenLemma.stream().filter(AutograderMain::isCorrect).count();
        double spellRatio = (1 - (double) correctCount / tokenLemma.size());
        List<Double> values = Arrays.asList(0D, 0.01, 0.022, 0.033, 0.088);
        return findIntervalIndex(spellRatio, values);

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

    /**
     * SUBJECT-VERB AGREEMENT
     **/
    private static boolean isSubjListSnglr(List<Integer> subjIndForVerbList, List<String> posList, List<String> wordList, int vi) {
        List<String> orNor = Arrays.asList("or", "nor");
        if (subjIndForVerbList.size() == 1) {
            return isSingleSubjSnglr(subjIndForVerbList, posList, wordList, 0);
        } else if (subjIndForVerbList.size() > 1) {
            int andCount = (int) IntStream.range(subjIndForVerbList.get(0), vi).filter(k -> wordList.get(k).equals("and")).count();
            int orNorCount = (int) IntStream.range(subjIndForVerbList.get(0), vi).filter(k -> orNor.contains(wordList.get(k))).count();
            if (andCount >= 1) {
                return false;
            } else if (orNorCount >= 1) {
                return isSingleSubjSnglr(subjIndForVerbList, posList, wordList, subjIndForVerbList.size() - 1);
            }
        }
        return false;//Find the success

    }

    private static boolean isValidInfVerbForm(List<String> infVerbPrecedesList, List<String> posList, int vi, SemanticGraph dependencyParse) {
        boolean ccomp = false;
        boolean dobj = false;
        for (TypedDependency t : dependencyParse.typedDependencies()) {
            if (t.dep().index() - 1 == vi) {
                if (t.reln().toString().equals("ccomp")) {// Mark helped his friend eat.
                    ccomp = true;
                }
                if (t.reln().toString().equals("dobj")) {//But first let us specify what (cats/cat) (eat/eats).
                    dobj = true;
                }
            }
        }
        if (ccomp) {
            return !dobj; // ccomp and dobj is not valid, ccomp alone is valid
        }
        return vi == 0 || infVerbPrecedesList.contains(posList.get(vi - 1)); // VB at index 0 is valid
    }

    private static boolean isSingleSubjSnglr(List<Integer> subjIndForVerbList, List<String> posList, List<String> wordList, int i) {
        List<String> nounSgPos = Arrays.asList("NNP", "NN");
        List<String> thirdPersSg = Arrays.asList("He", "She", "It", "he", "she", "it");
        List<String> demonsDetSg = Arrays.asList("This", "That", "this", "that");
        if (nounSgPos.contains(posList.get(subjIndForVerbList.get(i)))) {
            return true;
        } else if (posList.get(subjIndForVerbList.get(i)).equals("PRP")) {
            return thirdPersSg.contains(wordList.get(subjIndForVerbList.get(i)));
        } else if (posList.get(subjIndForVerbList.get(0)).equals("DT")) {
            return demonsDetSg.contains(wordList.get(subjIndForVerbList.get(0)));
        } else
            return posList.get(subjIndForVerbList.get(0)).equals("CD") && wordList.get(subjIndForVerbList.get(0)).toLowerCase().equals("one");
    }

    private static int getSubjectVerbAgrmntScore(Annotation document) {
        List<String> verbPos = Arrays.asList("VB", "VBP", "VBZ");
        List<String> infVerbPrecedesList = Arrays.asList("MD", "TO");
        int mistakeCount = 0;
        List<String> docTokenList = document.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.TextAnnotation.class)).collect(Collectors.toList());
        for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
            List<String> wordList = sentence.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.TextAnnotation.class)).collect(Collectors.toList());
            List<String> posList = sentence.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.PartOfSpeechAnnotation.class)).collect(Collectors.toList());
            List<Integer> verbIndexList = IntStream.range(0, posList.size()).filter(i -> verbPos.contains(posList.get(i))).boxed().collect(Collectors.toList());
            Map<Integer, Set<Integer>> verbSubjSetMap = new HashMap<>();
            SemanticGraph dependencyParse = sentence.get(SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation.class);
            for (int verbIndex : verbIndexList) {
                Set<Integer> subjIndForVerbSet = new HashSet<>();
                for (TypedDependency t : dependencyParse.typedDependencies()) {
                    if (t.gov().index() - 1 == verbIndex) {
                        if (t.reln().toString().equals("nsubj") && !t.dep().tag().equals("JJ")) {
                            subjIndForVerbSet.add(t.dep().index() - 1);
                        }
                    }
                }
                if (subjIndForVerbSet.isEmpty()) {
                    for (TypedDependency t : dependencyParse.typedDependencies()) {
                        if (t.dep().index() - 1 == verbIndex) {
                            if (t.reln().toString().equals("cop") || t.reln().toString().equals("aux")) {
                                for (TypedDependency c : dependencyParse.typedDependencies()) {
                                    if (c.gov().index() - 1 == t.gov().index() - 1) {
                                        if (c.reln().toString().equals("nsubj") && !c.dep().tag().equals("JJ")) {
                                            subjIndForVerbSet.add(c.dep().index() - 1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (!subjIndForVerbSet.isEmpty()) {
                    verbSubjSetMap.put(verbIndex, subjIndForVerbSet);
                }
            }

            for (Map.Entry<Integer, Set<Integer>> entry : verbSubjSetMap.entrySet()) {
                int vi = entry.getKey();
                List<Integer> subjIndForVerbList = new ArrayList<>(entry.getValue());
                Collections.sort(subjIndForVerbList);
                switch (posList.get(vi)) {
                    case "VBP":
                        if (isSubjListSnglr(subjIndForVerbList, posList, wordList, vi)) {
                            mistakeCount++;
                        }
                        break;
                    case "VBZ":
                        if (!isSubjListSnglr(subjIndForVerbList, posList, wordList, vi)) {
                            mistakeCount++;
                        }
                        break;
                    case "VB":
                        if (!isValidInfVerbForm(infVerbPrecedesList, posList, vi, dependencyParse)) {
                            if (isSubjListSnglr(subjIndForVerbList, posList, wordList, vi)) {
                                mistakeCount++;
                            }
                        }
                        break;
                }
            }
        }
        double subjVerbPercent = (1 - (((double) mistakeCount) / docTokenList.size())) * 100;
        List<Double> values = Arrays.asList(0.0, 97.70, 98.71, 99.28, 99.78);
        return findIntervalIndex(subjVerbPercent, values) + 1;
    }

    /**
     * MAIN
     **/
    public static void main(String[] args) {
        // TODO train/test switch from args
        trainGrader();
        //testGrader();
    }

    private static void trainGrader() {
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
                int spellScore = spellCheck(document);
                int subjVerbAgrmntScore = getSubjectVerbAgrmntScore(document);
                System.out.println(nextRecord[0] + "\t" + lengthScore + "\t" + spellScore + "\t" + subjVerbAgrmntScore + "\t" + nextRecord[2]);
                // TODO find best weights for features using linear regression
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void testGrader() {
        try {
            Reader reader = Files.newBufferedReader(Paths.get("input/testing/index.csv"));
            CSVParser csvParser = new CSVParserBuilder().withSeparator(';').build();
            CSVReader csvReader = new CSVReaderBuilder(reader).withCSVParser(csvParser).withSkipLines(1).build();
            String[] nextRecord;

            Properties props = new Properties();
            props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
            StanfordCoreNLP pipeline = new StanfordCoreNLP(props);


            while ((nextRecord = csvReader.readNext()) != null) {
                BufferedReader essayReader = Files.newBufferedReader(Paths.get("input/testing/essays/" + nextRecord[0]));
                StringBuilder essay = new StringBuilder();
                String line;
                while ((line = essayReader.readLine()) != null) {
                    essay.append(line).append("\n");
                }

                Annotation document = new Annotation(essay.toString());
                pipeline.annotate(document);
                int lengthScore = getLengthScore(document);
                int spellScore = spellCheck(document);
                int subjVerbAgrmntScore = getSubjectVerbAgrmntScore(document);
                System.out.println(nextRecord[0] + "\t" + lengthScore + "\t" + spellScore + "\t" + subjVerbAgrmntScore + "\t" + nextRecord[2]);
                // TODO get final score
                // TODO write output to "output/results.txt"
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * DEBUG
     **/
    public static void main1(String[] args) throws IOException {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
//        String doc = "John or Jane eat food.";
//        String doc = "Another is on the way.";
//        String doc = "Anyone who sees his or her friends runs to greet them.";
//        String doc = "Give him an ornament that he eat.";
        Reader reader = Files.newBufferedReader(Paths.get("input/training/index.csv"));
        CSVParser csvParser = new CSVParserBuilder().withSeparator(';').build();
        CSVReader csvReader = new CSVReaderBuilder(reader).withCSVParser(csvParser).withSkipLines(1).build();
        String[] nextRecord;
//        BufferedReader essayReader = Files.newBufferedReader(Paths.get("input/training/essays/937403.txt"));
        while ((nextRecord = csvReader.readNext()) != null) {
            BufferedReader essayReader = Files.newBufferedReader(Paths.get("input/training/essays/" + nextRecord[0]));
            StringBuilder essay = new StringBuilder();
            String line;
            while ((line = essayReader.readLine()) != null) {
                essay.append(line).append("\n");
            }
//        Annotation document = new Annotation(doc);
            Annotation document = new Annotation(essay.toString());
            pipeline.annotate(document);
//            System.out.println(document);
            List<CoreMap> sentenceList = document.get(CoreAnnotations.SentencesAnnotation.class);
            for (CoreMap sentence : sentenceList) {
//                SemanticGraph dependencyParse = sentence.get(SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation.class);
//                for (TypedDependency t : dependencyParse.typedDependencies()) {
//                    System.out.println(t.gov() + " " + t.reln() + " " + t.dep() + " " + t.dep().index());
//                }
                if (sentence.toString().contains("?")) {
                    System.out.println(sentence);
                }
            }
        }
//        List<Double> values = Arrays.asList(0D, 1.2, 2D, 3.5, 4.8);
//        System.out.println(findIntervalIndex(3.4, values));
//        System.out.println(getSubjectVerbAgrmntScore(document));

    }


}
