import com.opencsv.*;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.IDictionary;
import edu.mit.jwi.item.*;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.CoreMap;

import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.Sets;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class AutograderMain {

    private static IDictionary dictionary = null;
    private static HashSet<String> closedWords_en = null;
    private static HashSet<String> stopwords_en = null;
    private static String resPathPrefix = "";
    private static String ioPathPrefix = "../";
    private static List<String> sbarParents = Arrays.asList("S", "SINV", "VP", "NP");
    private static List<String> sbarChildren = Arrays.asList("IN", "WHNP", "WHPP", "WHADJP", "WHADVP", "S");
    private static List<String> sentStartConflictVerbs = Arrays.asList("VB", "VBN", "VBZ", "VBP", "VBD");
    private static List<String> tagsToExclude = Arrays.asList(",", ".", "``", "''", ":", "#", "", "--", "$", "-NONE-", "-LRB-", "-RRB-", "POS");
    private static Set<String> allTreebankRules = new HashSet<>();
    private static Map<String, Integer> allSeqMistakesFreq = new HashMap<>();
    private static Set<String> allSeqMistakes = new HashSet<>();

    static {
        String execPath = System.getProperty("user.dir");
        if (!execPath.contains("executable")) {
            resPathPrefix = "executable/";
            ioPathPrefix = "";
        }
        try {
            closedWords_en = new HashSet<>(Files.readAllLines(Paths.get(resPathPrefix + "resources/libs/closed_class.txt")));
            stopwords_en = new HashSet<>(Files.readAllLines(Paths.get(resPathPrefix + "resources/libs/stopwords.txt")));
        } catch (IOException e) {
            e.printStackTrace();
        }
        URL url = null;
        try {
            url = new URL("file", null, resPathPrefix + "resources/libs/dict");
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


    static int getGrammarScore(Annotation document) {
        double badScore = 0;
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            boolean hasSubject = containsSubject(sentence);
            boolean hasVerb = containsVerb(sentence);
            int posSequenceErrors = computePosSequenceErrors(sentence);
            if (!hasVerb) badScore += 2;
            if (!hasSubject) badScore += 1;
            badScore += 5 * posSequenceErrors;
        }
        double normalizedScore = badScore / sentences.size();
        List<Double> values = Arrays.asList(0D, 0.19D, 0.37, 0.62, 1.77);
        return 5 - findIntervalIndex(normalizedScore, values); //5 minus is done since we return correctness score and normalized score is wrongness score.
    }

    private static int computePosSequenceErrors(CoreMap sentence) {
        return BadPosSequence.getBadSequenceCount(sentence);
    }


    private static boolean containsVerb(CoreMap sentence) {
        Class<CoreAnnotations.TokensAnnotation> tokenType = CoreAnnotations.TokensAnnotation.class;
        Class<CoreAnnotations.PartOfSpeechAnnotation> posType = CoreAnnotations.PartOfSpeechAnnotation.class;
        Function<CoreLabel, String> tokenToPos = token -> token.get(posType);
        return sentence.get(tokenType).stream().map(tokenToPos).anyMatch(pos -> pos.contains("VB"));
    }

    private static boolean containsSubject(CoreMap sentence) {
        boolean hasSubject = false;
        SemanticGraph dependencyParse = sentence.get(SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation.class);
        for (TypedDependency t : dependencyParse.typedDependencies()) {
            if (t.reln().toString().contains("subj")) {
                hasSubject = true;
                break;
            }
        }
        return hasSubject;
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

    private static void traverseParseTree(Tree child, Tree root, Map<String, Integer> sfAttrCountsMap, Set<String> allParentChildren) {
        if (sfAttrCountsMap.get("FRAGX") == 0) {
            if ("FRAG".equals(child.label().value()) || "X".equals(child.label().value())) {
                sfAttrCountsMap.put("FRAGX", 1);
                return;
            } else if ("S".equals(child.label().value())) {
                sfAttrCountsMap.put("S", sfAttrCountsMap.get("S") + 1);
            } else if ("SBAR".equals(child.label().value())) {
                if (!(sbarParents.contains(child.parent(root).value())) || child.getChildrenAsList().stream().noneMatch(node -> sbarChildren.contains(node.label().value()))) {
                    sfAttrCountsMap.put("SBAR", sfAttrCountsMap.get("SBAR") + 1);
                }
            }
        }

        extractAllParentChildSeqs(child, allParentChildren);

        for (Tree each : child.children()) {
            traverseParseTree(each, root, sfAttrCountsMap, allParentChildren);
        }

    }

    private static void extractAllParentChildSeqs(Tree child, Set<String> allParentChildren) {
        if (isValidSentFormNode(child)) {
            StringBuilder pcr = new StringBuilder();
            List<String> children = child.getChildrenAsList().stream().filter(c -> isValidTag(c.label().value())).map(c -> c.label().value()).collect(Collectors.toList());
            if (children.size() > 0) {
                pcr.append(child.label().value().split("-")[0]).append(",");
                String s = children.stream().reduce((catStr1, catStr2) -> catStr1.split("-")[0] + "," + catStr2.split("-")[0]).get();
                pcr.append(s);
                allParentChildren.add(pcr.toString());
            }
        }
    }

    private static boolean isValidTag(String tag) {
        return !tagsToExclude.contains(tag);
    }

    private static boolean isValidSentFormNode(Tree parent) {
        boolean isValid = false;
        if (!parent.isLeaf() && !Arrays.asList(new String[]{"ROOT", "FRAG", "X", "SBAR"}).contains(parent.label().value())) {
            isValid = true;
            for (Tree child : parent.getChildrenAsList()) {
                if (child.isLeaf() || Arrays.asList(new String[]{"ROOT", "FRAG", "X", "SBAR"}).contains(child.label().value())) {
                    isValid = false;
                }
            }


        }
        return isValid;
    }

    static int getSentenceFormationScore(Annotation document, String grade) {
        int essayPenalty = 0;
        int numWrongSents = 0;
        int totalNumSents = 0;
        for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
            totalNumSents++;
            int fragxPenalty = 0;
            int clausePenalty = 0;
            int sbarPenalty = 0;
            int startVerbPenalty = 0;
            int missingWordsConstPenalty = 0;
            Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
            Map<String, Integer> sfAttrCountsMap = new HashMap<>();
            Set<String> allParentChildren = new HashSet<>();
            sfAttrCountsMap.put("FRAGX", 0);
            sfAttrCountsMap.put("S", 0);
            sfAttrCountsMap.put("SBAR", 0);
            traverseParseTree(tree, tree, sfAttrCountsMap, allParentChildren);
            List<String> posList = sentence.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.PartOfSpeechAnnotation.class)).collect(Collectors.toList());

            if (sfAttrCountsMap.get("FRAGX") == 1) {
                fragxPenalty = 5;
            } else {
                if (sentStartConflictVerbs.contains(posList.get(0))) {
                    startVerbPenalty = 2;
                }
                if (sfAttrCountsMap.get("S") == 0) {
                    clausePenalty = 3;
                }
                if (sfAttrCountsMap.get("SBAR") > 0) {
                    sbarPenalty = 2 * sfAttrCountsMap.get("SBAR");
                }
                missingWordsConstPenalty = getMissingWordsConstPenalty(allParentChildren, grade);
            }
            int totalPenalty = fragxPenalty + clausePenalty + sbarPenalty + startVerbPenalty + missingWordsConstPenalty;
            essayPenalty += totalPenalty;
            if (totalPenalty > 0) {
                numWrongSents++;
            }

        }
        double sentFormMistakes = (numWrongSents / (double) totalNumSents);
        List<Double> values = Arrays.asList(0.0, 0.418, 0.485, 0.660, 0.902);
        return 5 - findIntervalIndex(sentFormMistakes, values);
    }

    private static int getMissingWordsConstPenalty(Set<String> allParentChildren, String grade) {
        int missingWordConstCount = 0;
        try {
                BufferedReader rulesReader = Files.newBufferedReader(Paths.get(resPathPrefix + "resources/treebank_rules.txt"));
                String nextLine;
                while ((nextLine = rulesReader.readLine()) != null) {
                    allTreebankRules.add(nextLine);
                }
                /*List<String> mistakesList = allParentChildren.stream().filter(seq -> !allTreebankRules.contains(seq) && grade.equals("high")).collect(Collectors.toList());
                for (String m : mistakesList) {
                    allSeqMistakesFreq.put(m, allSeqMistakesFreq.getOrDefault(m, 0) + 1);
                }*/
                missingWordConstCount = (int) allParentChildren.stream().filter(seq -> !allTreebankRules.contains(seq)).count();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return missingWordConstCount;
    }

    /**
     * MAIN
     **/
    // TODO add comments
    public static void main(String[] args) {
        boolean rebuild = false;
        if (args.length > 1) {
            rebuild = Boolean.valueOf(args[1]);
        }
        Map<String, Consumer> tasks = new HashMap<>();
        tasks.put("test", bool -> testGrader());
        tasks.put("train", (Consumer<Boolean>) AutograderMain::trainGrader);
        tasks.get(args[0]).accept(rebuild);
//        testGrader();
    }

    private static void trainGrader(boolean buildFeatures) {
        if (buildFeatures) {
            try {
                Reader reader = Files.newBufferedReader(Paths.get(ioPathPrefix + "input/training/index.csv"));
                CSVParser csvParser = new CSVParserBuilder().withSeparator(';').build();
                CSVReader csvReader = new CSVReaderBuilder(reader).withCSVParser(csvParser).withSkipLines(1).build();
                String[] nextRecord;

                Properties props = new Properties();
                props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
                StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

                Writer writer = Files.newBufferedWriter(Paths.get(resPathPrefix + "resources/train_features.csv"));

                CSVWriter csvWriter = new CSVWriter(writer,
                        CSVWriter.DEFAULT_SEPARATOR,
                        CSVWriter.NO_QUOTE_CHARACTER,
                        CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                        CSVWriter.DEFAULT_LINE_END);
                String[] headerRecord = {"File", "a", "b", "c_i", "c_ii", "c_iii", "d_i", "d_ii", "class"};
                csvWriter.writeNext(headerRecord);

                while ((nextRecord = csvReader.readNext()) != null) {
                    BufferedReader essayReader = Files.newBufferedReader(Paths.get(ioPathPrefix + "input/training/essays/" + nextRecord[0]));
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
                    int grammarScore = getGrammarScore(document);
                    int sentFormScore = getSentenceFormationScore(document, nextRecord[2]);
                    int topicScore = getTopicRelevanceScore(document, nextRecord[1]);


                    System.out.println(nextRecord[0] + "\t" + lengthScore + "\t" + spellScore + "\t" + subjVerbAgrmntScore + "\t" + grammarScore + "\t" + topicScore + "\t" + sentFormScore + "\t" + nextRecord[2]);

                    csvWriter.writeNext(new String[]{nextRecord[0], String.valueOf(lengthScore), String.valueOf(spellScore), String.valueOf(subjVerbAgrmntScore), String.valueOf(grammarScore), String.valueOf(topicScore), String.valueOf(sentFormScore), String.valueOf(0), String.valueOf(0), nextRecord[2]});
                    essayReader.close();

                }
                /*for(Map.Entry<String, Integer> entry: allSeqMistakesFreq.entrySet()){
                    if(entry.getValue() > 2){
                        allSeqMistakes.add(entry.getKey());
                    }
                }*/
                writer.close();
                reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {

            Instances trainingDataset = getDataSet(resPathPrefix + "resources/train_features.csv");
            Classifier classifier = new weka.classifiers.functions.SMO();
            ((SMO) classifier).setOptions(weka.core.Utils.splitOptions("-C 1 -N 2"));
            classifier.buildClassifier(trainingDataset);
            System.out.println(classifier);

            Evaluation eval = new Evaluation(trainingDataset);
//            Instances testingDataSet = getDataSet("executable/resources/predict_data_set.csv");
//            eval.evaluateModel(classifier, testingDataSet);
            eval.crossValidateModel(classifier, trainingDataset, 10, new Random(1));
            System.out.println(eval.toSummaryString());
            Instances predictDataset = getDataSet(resPathPrefix + "resources/predict_data_set.csv");
            for (Instance i : predictDataset) {
                double value = classifier.classifyInstance(i);
                if (i.classValue() != value) {
                    System.out.println(i);
                    System.out.println(value);
                }

            }

            weka.core.SerializationHelper.write(resPathPrefix + "resources/essay_grader.model", classifier);
        } catch (Exception e) {
            e.printStackTrace();
        }


    }


    private static Instances getDataSet(String filePath) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filePath));
        Instances dataset = loader.getDataSet();

        dataset.deleteAttributeAt(0);
        dataset.setClassIndex(7);

        return dataset;
    }

    //TODO update function, grade
    private static void testGrader() {
        try {
            Reader reader = Files.newBufferedReader(Paths.get(ioPathPrefix + "input/testing/index.csv"));
            CSVParser csvParser = new CSVParserBuilder().withSeparator(';').build();
            CSVReader csvReader = new CSVReaderBuilder(reader).withCSVParser(csvParser).withSkipLines(1).build();
            String[] nextRecord;

            Properties props = new Properties();
            props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
            StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

            Writer writer = Files.newBufferedWriter(Paths.get(ioPathPrefix + "output/results.txt"));

            while ((nextRecord = csvReader.readNext()) != null) {
                BufferedReader essayReader = Files.newBufferedReader(Paths.get(ioPathPrefix + "input/testing/essays/" + nextRecord[0]));
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
                int grammarScore = getGrammarScore(document);
                int topicScore = getTopicRelevanceScore(document, nextRecord[1]);
                double finalScore = 2.1429 * lengthScore - 0.8571 * spellScore - 0.1429 * subjVerbAgrmntScore * 0.2857 * grammarScore;
                String finalGrade = "unknown";
                System.out.println(nextRecord[0] + ";" + lengthScore + ";" + spellScore + ";" + subjVerbAgrmntScore + ";" + grammarScore + ";"  + topicScore + ";" + (int) finalScore + ";" + finalGrade);
                String scoreDetails = nextRecord[0] + ";" + lengthScore + ";" + spellScore + ";" + subjVerbAgrmntScore + ";" + grammarScore + ";"  + topicScore + 0 + ";" + 0 + ";" + (int) finalScore + ";" + finalGrade + "\n";
                writer.write(scoreDetails);
                essayReader.close();
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private static int getTopicRelevanceScore(Annotation document, String topic) throws IOException {
        if(!dictionary.isOpen()) dictionary.open();
        if(topic.contains("\t")) topic = topic.split("\t+")[1];
        Annotation topicAnnotation = processTopic(topic);
        Set<String> topicNouns = getMainNouns(topicAnnotation);
        Set<String> documentNouns = getMainNouns(document);
        Set<LinkedList<ISynset>> topicHyperTrees = topicNouns.stream().flatMap(word -> findRelationalTrees(word, AutograderMain::getHypernyms).stream()).collect(Collectors.toSet());
        Map<String, Double> wordScores = documentNouns.stream().collect(Collectors.toMap(Function.identity(), word -> getSimilarityScore(word, topicHyperTrees)));
        dictionary.close();
        if(wordScores.isEmpty()) return 0;
        return ((Double)wordScores.values().stream().mapToDouble(value -> value).average().getAsDouble()).intValue();
    }

    private static Double getSimilarityScore(String word, Set<LinkedList<ISynset>> topicHyperTrees) {
        Set<LinkedList<ISynset>> wordHyperTrees = findRelationalTrees(word, AutograderMain::getHypernyms);
        if(!topicHyperTrees.isEmpty() && !wordHyperTrees.isEmpty()) {
            Double score = Sets.cross(wordHyperTrees, topicHyperTrees).stream().map(treePair -> getSimilarityScore(treePair.first, treePair.second)).max(Comparator.naturalOrder()).get();
            return score;
        }
        return 0D;
    }

    private static Double getSimilarityScore(LinkedList<ISynset> wordTree, LinkedList<ISynset> topicTree) {
        Double max_score = 3D;
        if(topicTree.getLast().equals(wordTree.getLast())) return max_score;
        double score = 0D;
        Set<ISynset> wordNetTopic = wordTree.getLast().getRelatedMap().getOrDefault(Pointer.TOPIC, Generics.newArrayList()).stream().map(id -> dictionary.getSynset(id)).collect(Collectors.toSet());
        if(Sets.intersects(wordNetTopic, new HashSet<>(topicTree))) score += max_score;

        if(wordTree.contains(topicTree.getLast()) || topicTree.contains(wordTree.getLast())) score += max_score;

        Set<ISynset> commonNodes = Sets.intersection(new HashSet<>(wordTree), new HashSet<>(topicTree));
        if(commonNodes.isEmpty()) return score;
        ISynset lowestCommonNode = commonNodes.stream().max(Comparator.comparingInt(wordTree::indexOf)).get();
        int depth1 = wordTree.indexOf(lowestCommonNode) + 1;
        int depth2 = topicTree.indexOf(lowestCommonNode) + 1;
        int depth = depth1 < depth2 ? depth1 : depth2;
        score += (double) (6 * depth) / (wordTree.size() + topicTree.size());

        return score < max_score ? score : max_score;
    }

    private static Set<LinkedList<ISynset>> findRelationalTrees(String word, Function<ISynset, Set<ISynset>> relationFunction) {
        return dictionary.getIndexWord(word, POS.NOUN).getWordIDs().stream()
                .map(wordID -> dictionary.getWord(wordID).getSynset())
                .flatMap(synset -> findRelationalTrees(synset, relationFunction).stream())
                .collect(Collectors.toSet());
    }

    private static Set<LinkedList<ISynset>> findRelationalTrees(ISynset word, Function<ISynset, Set<ISynset>> relationFunction) {
        Set<LinkedList<ISynset>> incompleteHyperTrees =  new HashSet<>(Collections.singletonList(new LinkedList<>(Collections.singletonList(word))));
        Set<LinkedList<ISynset>> completeHyperTrees =  new HashSet<>();

        while (!incompleteHyperTrees.isEmpty()) {
            LinkedList<ISynset> currentHyperTree = incompleteHyperTrees.iterator().next();
            incompleteHyperTrees.remove(currentHyperTree);
            ISynset head = currentHyperTree.getFirst();
            Set<ISynset> parents = relationFunction.apply(head);
            if(parents.isEmpty()) completeHyperTrees.add(currentHyperTree);
            Stream<LinkedList<ISynset>> updatedHyperTrees = parents.stream().map(parent -> Stream.concat(Stream.of(parent), currentHyperTree.stream()).collect(Collectors.toCollection(LinkedList::new)));
            updatedHyperTrees.forEach(incompleteHyperTrees::add);
        }
        return completeHyperTrees;
    }

    private static Set<ISynset> getHypernyms(ISynset head) {
        return Sets.union(new HashSet<>(head.getRelatedMap().getOrDefault(Pointer.HYPERNYM, new ArrayList<>())), new HashSet<>(head.getRelatedMap().getOrDefault(Pointer.HYPERNYM_INSTANCE, new ArrayList<>()))).stream().map(dictionary::getSynset).collect(Collectors.toSet());
    }

    private static Set<ISynset> getRelated(ISynset head, Pointer relation) {
        return new HashSet<>(head.getRelatedMap().getOrDefault(relation, new ArrayList<>())).stream().map(dictionary::getSynset).collect(Collectors.toSet());
    }

    private static Set<ISynset> getMeronyms(ISynset head) {
        Set<ISynsetID> partUnion = Sets.union(new HashSet<>(head.getRelatedMap().getOrDefault(Pointer.MERONYM_MEMBER, new ArrayList<>())), new HashSet<>(head.getRelatedMap().getOrDefault(Pointer.MERONYM_PART, new ArrayList<>())));
        return Sets.union(partUnion, new HashSet<>(head.getRelatedMap().getOrDefault(Pointer.MERONYM_SUBSTANCE, new ArrayList<>()))).stream().map(dictionary::getSynset).collect(Collectors.toSet());
    }

    private static Set<String> getMainNouns(Annotation document) {
        return document.get(CoreAnnotations.TokensAnnotation.class).stream()
                .filter(token -> token.get(CoreAnnotations.PartOfSpeechAnnotation.class).contains("NN"))
                .map(token -> token.get(CoreAnnotations.LemmaAnnotation.class))
                .filter(word -> !stopwords_en.contains(word.toLowerCase()) && dictionary.getIndexWord(word, POS.NOUN) != null)
                .collect(Collectors.toSet());
    }

    private static Annotation processTopic(String topic) {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        Annotation topicAnnotation = new Annotation(topic);
        pipeline.annotate(topicAnnotation);
        return topicAnnotation;
    }

}
