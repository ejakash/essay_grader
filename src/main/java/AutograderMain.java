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
//    private static Map<String, Integer> allSeqMistakesFreq = new HashMap<>();
//    private static Set<String> allSeqMistakes = new HashSet<>();

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

    /**
     * find the interval to index mapping, for converting range to score
     *
     * @param searchValue value to map
     * @param intervals   list of intervals
     * @param <N>         Number
     * @return mapped index
     */
    private static <N extends Number> int findIntervalIndex(Number searchValue, List<N> intervals) {
        for (int i = 0; i < intervals.size() - 1; i++) {
            if (intervals.get(i).doubleValue() <= searchValue.doubleValue() && intervals.get(i + 1).doubleValue() > searchValue.doubleValue())
                return i;
        }
        return intervals.size() - 1;
    }

    /**
     * part (a) - get the score based on length
     *
     * @param document annotated document
     * @return score
     */
    private static int getLengthScore(Annotation document) {
        int sentenceCount = 0;
        String[] sepArr = {"CC", "IN", ",", "WRB", "WDT", "WP", "WP$"};// all pos tags that separate two independent clauses
        for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
            SemanticGraph dependencyParse =
                    sentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);// get the dependency graph

            int sentenceSplitCount = 0;// count for possible sentences within a sentence
            List<Integer> subjIndexList = new ArrayList<>();// list of subject indices
            for (TypedDependency t : dependencyParse.typedDependencies()) {
                if (t.reln().toString().contains("subj")) {
                    String s = t.dep().originalText();
                    if (Character.isUpperCase(s.charAt(0))) {
                        subjIndexList.add(t.dep().index());// if relation is subject and the dependent starts with uppercase, add its index
                    }
                }
            }
            List<String> posList = sentence.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.PartOfSpeechAnnotation.class)).collect(Collectors.toList());// get list of pos tags
            Collections.sort(subjIndexList);// sort indices
            if (subjIndexList.size() > 1) {
                for (int i = 1; i < subjIndexList.size(); i++) {
                    int sepCount = (int) IntStream.range(subjIndexList.get(i - 1), subjIndexList.get(i))
                            .filter(k -> (Arrays.asList(sepArr).contains(posList.get(k - 1)))).count();// count separators between clauses
                    if (sepCount == 0) {
                        sentenceSplitCount++;// increment number of possible sentences, if there are no separators
                    }
                }
            }
            if (sentenceSplitCount > 1) {
                sentenceCount += sentenceSplitCount;// update total number of sentences
            } else {
                sentenceCount++;
            }
        }

        List<Integer> values = Arrays.asList(0, 10, 13, 16, 20);// thresholds based on mean and standard deviation
        return findIntervalIndex(sentenceCount, values) + 1;// return 1-5 mapped score

    }

    /**
     * part (b) - get the score based on misspells
     *
     * @param document annotated document
     * @return score
     */
    private static int spellCheck(Annotation document) {
        List<String> tokenLemma = document.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.LemmaAnnotation.class)).collect(Collectors.toList());// get lemmatized tokens
        Long correctCount = tokenLemma.stream().filter(AutograderMain::isCorrect).count();// get the count of correct words
        double spellRatio = (1 - (double) correctCount / tokenLemma.size());// get the ratio of misspells
        List<Double> values = Arrays.asList(0D, 0.01, 0.022, 0.033, 0.088);// thresholds based on mean and standard deviation
        return findIntervalIndex(spellRatio, values);// return 0-4 mapped score

    }

    /**
     * check for correct spelling
     *
     * @param text word
     * @return boolean
     */
    private static boolean isCorrect(String text) {
        if (closedWords_en == null) System.out.println("Unable to read the closed word list");
        else if (closedWords_en.contains(text.toLowerCase())) return true;
        if (dictionary == null) {
            System.out.println("Unable to read the wordnet dictionary");
            //Assume all words to the true if the spelling checker fails to load
            return true;
        }
        try {
            dictionary.open();// open wordnet dictionary
        } catch (IOException e) {
            e.printStackTrace();
        }
        POS[] posList = {POS.NOUN, POS.VERB, POS.ADJECTIVE, POS.ADVERB};// wordnet pos tags
        for (POS pos : posList) {
            IIndexWord idxWord = dictionary.getIndexWord(text, pos);// search in dictionary
            if (idxWord != null) return true;
        }
        dictionary.close();
        return false;
    }

    /**
     * part (c ii) - get verb mistakes score
     *
     * @param document annotated document
     * @return score
     */
    static int getGrammarScore(Annotation document) {
        double badScore = 0;
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);// get sentences
        for (CoreMap sentence : sentences) {
            boolean hasSubject = containsSubject(sentence);// check if subject is present, for main verb presence
            boolean hasVerb = containsVerb(sentence);// check if any verb is present
            int posSequenceErrors = computePosSequenceErrors(sentence);// count rule violations for verb tense and other missing/extra verbs
            if (!hasVerb) badScore += 2;
            if (!hasSubject) badScore += 1;
            badScore += 5 * posSequenceErrors;
        }
        double normalizedScore = badScore / sentences.size();
        List<Double> values = Arrays.asList(0D, 0.19D, 0.37, 0.62, 1.77);// thresholds based on mean and standard deviation
        return 5 - findIntervalIndex(normalizedScore, values);// 5 minus is done since we return correctness score and normalized score is wrongness score.
    }

    /**
     * compute all pos sequence errors
     *
     * @param sentence sentence
     * @return count
     */
    private static int computePosSequenceErrors(CoreMap sentence) {
        return BadPosSequence.getBadSequenceCount(sentence);
    }

    /**
     * check if sentence contains any verb
     *
     * @param sentence sentence
     * @return boolean
     */
    private static boolean containsVerb(CoreMap sentence) {
        Class<CoreAnnotations.TokensAnnotation> tokenType = CoreAnnotations.TokensAnnotation.class;
        Class<CoreAnnotations.PartOfSpeechAnnotation> posType = CoreAnnotations.PartOfSpeechAnnotation.class;
        Function<CoreLabel, String> tokenToPos = token -> token.get(posType);
        return sentence.get(tokenType).stream().map(tokenToPos).anyMatch(pos -> pos.contains("VB"));
    }

    /**
     * check if sentence contains a subject
     *
     * @param sentence sentence
     * @return boolean
     */
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
     * check if the subject(s) of the verb at index vi, is singular
     *
     * @param subjIndForVerbList list of subject indices
     * @param posList            list of pos tags
     * @param wordList           list of words
     * @param vi                 verb index
     * @return boolean
     */
    private static boolean isSubjListSnglr(List<Integer> subjIndForVerbList, List<String> posList, List<String> wordList, int vi) {
        List<String> orNor = Arrays.asList("or", "nor");
        if (subjIndForVerbList.size() == 1) {// if there is only one entity in the subject
            return isSingleSubjSnglr(subjIndForVerbList, posList, wordList, 0);// check if the single subject is singular
        } else if (subjIndForVerbList.size() > 1) {// if there are multiple entities in the subject
            int andCount = (int) IntStream.range(subjIndForVerbList.get(0), vi).filter(k -> wordList.get(k).equals("and")).count();// count the entities separated by and
            int orNorCount = (int) IntStream.range(subjIndForVerbList.get(0), vi).filter(k -> orNor.contains(wordList.get(k))).count();// count the enitities separated by or/nor
            if (andCount >= 1) {
                return false;// plural if there is atleast one 'and'
            } else if (orNorCount >= 1) {// if there is atleast one or/nor
                return isSingleSubjSnglr(subjIndForVerbList, posList, wordList, subjIndForVerbList.size() - 1);// check if the last entity is singular
            }
        }
        return false;// default to plural <Ex: Find the success.>

    }

    /**
     * check if the infinitival verb form (VB) is valid
     *
     * @param infVerbPrecedesList list of infinitival verb preceding tags
     * @param posList             list of pos tags
     * @param vi                  verb index
     * @param dependencyParse     dependency graph
     * @return boolean
     */
    private static boolean isValidInfVerbForm(List<String> infVerbPrecedesList, List<String> posList, int vi, SemanticGraph dependencyParse) {
        boolean ccomp = false;// clausal complement
        boolean dobj = false;// direct object
        for (TypedDependency t : dependencyParse.typedDependencies()) {
            if (t.dep().index() - 1 == vi) {
                if (t.reln().toString().equals("ccomp")) {// <Ex:Mark helped his friend eat.>
                    ccomp = true;
                }
                if (t.reln().toString().equals("dobj")) {// <Ex: But first let us specify what (cats/cat) (eat/eats).>
                    dobj = true;
                }
            }
        }
        if (ccomp) {
            return !dobj; // ccomp and dobj is not valid, ccomp alone is valid
        }
        return vi == 0 || infVerbPrecedesList.contains(posList.get(vi - 1)); // VB at index 0 is valid
    }

    /**
     * check if single subject is singular
     *
     * @param subjIndForVerbList list of subject indices
     * @param posList            list of pos tags
     * @param wordList           lit of words
     * @param i                  index
     * @return boolean
     */
    private static boolean isSingleSubjSnglr(List<Integer> subjIndForVerbList, List<String> posList, List<String> wordList, int i) {
        List<String> nounSgPos = Arrays.asList("NNP", "NN"); // singular nouns
        List<String> thirdPersSg = Arrays.asList("He", "She", "It", "he", "she", "it");// third person singular
        List<String> demonsDetSg = Arrays.asList("This", "That", "this", "that");// singular demonstrative determiners
        if (nounSgPos.contains(posList.get(subjIndForVerbList.get(i)))) {
            return true;
        } else if (posList.get(subjIndForVerbList.get(i)).equals("PRP")) {
            return thirdPersSg.contains(wordList.get(subjIndForVerbList.get(i)));// check if PRP is singular
        } else if (posList.get(subjIndForVerbList.get(0)).equals("DT")) {
            return demonsDetSg.contains(wordList.get(subjIndForVerbList.get(0)));// check if DT is singular
        } else
            return posList.get(subjIndForVerbList.get(0)).equals("CD") && wordList.get(subjIndForVerbList.get(0)).toLowerCase().equals("one");// if CD is the word 'one'
    }

    /**
     * part (c i) - get the subject-verb agreement score
     *
     * @param document annotated document
     * @return score
     */
    private static int getSubjectVerbAgrmntScore(Annotation document) {
        List<String> verbPos = Arrays.asList("VB", "VBP", "VBZ");// possible verbs that change with subject number
        List<String> infVerbPrecedesList = Arrays.asList("MD", "TO");// tags preceding infinitival verb(VB)
        int mistakeCount = 0;
        List<String> docTokenList = document.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.TextAnnotation.class)).collect(Collectors.toList());// get tokens for whole essay
        for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
            List<String> wordList = sentence.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.TextAnnotation.class)).collect(Collectors.toList());// get words for this sentence
            List<String> posList = sentence.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.PartOfSpeechAnnotation.class)).collect(Collectors.toList());// get pos list
            List<Integer> verbIndexList = IntStream.range(0, posList.size()).filter(i -> verbPos.contains(posList.get(i))).boxed().collect(Collectors.toList());// get all verb indices
            Map<Integer, Set<Integer>> verbSubjSetMap = new HashMap<>();// verb index to subject indices set map
            SemanticGraph dependencyParse = sentence.get(SemanticGraphCoreAnnotations.EnhancedPlusPlusDependenciesAnnotation.class);// get dependency graph
            for (int verbIndex : verbIndexList) {
                Set<Integer> subjIndForVerbSet = new HashSet<>();
                for (TypedDependency t : dependencyParse.typedDependencies()) {
                    if (t.gov().index() - 1 == verbIndex) {// if verb is the governor
                        if (t.reln().toString().equals("nsubj") && !t.dep().tag().equals("JJ")) {
                            subjIndForVerbSet.add(t.dep().index() - 1);// get all the subject indices for dependents that are not adjectives
                        }
                    }
                }
                if (subjIndForVerbSet.isEmpty()) {// if subject list is empty, check for verb as dependent
                    for (TypedDependency t : dependencyParse.typedDependencies()) {
                        if (t.dep().index() - 1 == verbIndex) {// if verb is the dependent
                            if (t.reln().toString().equals("cop") || t.reln().toString().equals("aux")) {// check if the verb is copular or auxiliary
                                for (TypedDependency c : dependencyParse.typedDependencies()) {
                                    if (c.gov().index() - 1 == t.gov().index() - 1) { // get the subject from the governor where the dependent was the verb <Ex: Ivan is the best dancer.>
                                        if (c.reln().toString().equals("nsubj") && !c.dep().tag().equals("JJ")) {
                                            subjIndForVerbSet.add(c.dep().index() - 1);// get the subject idices for dependents that are not adjectives
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (!subjIndForVerbSet.isEmpty()) {
                    verbSubjSetMap.put(verbIndex, subjIndForVerbSet);// update the map
                }
            }

            for (Map.Entry<Integer, Set<Integer>> entry : verbSubjSetMap.entrySet()) {
                int vi = entry.getKey();// verb index
                List<Integer> subjIndForVerbList = new ArrayList<>(entry.getValue());// list of subjects
                Collections.sort(subjIndForVerbList);// sort list of subjects
                switch (posList.get(vi)) {
                    case "VBP":// if VBP and subject is singular, count as mistake
                        if (isSubjListSnglr(subjIndForVerbList, posList, wordList, vi)) {
                            mistakeCount++;
                        }
                        break;
                    case "VBZ":// if VBZ and subject is plural, count as mistake
                        if (!isSubjListSnglr(subjIndForVerbList, posList, wordList, vi)) {
                            mistakeCount++;
                        }
                        break;
                    case "VB":// if VB is invalid, if subject is singular, count as mistake
                        if (!isValidInfVerbForm(infVerbPrecedesList, posList, vi, dependencyParse)) {
                            if (isSubjListSnglr(subjIndForVerbList, posList, wordList, vi)) {
                                mistakeCount++;
                            }
                        }
                        break;
                }
            }
        }
        double subjVerbPercent = (1 - (((double) mistakeCount) / docTokenList.size())) * 100;// get the subject-verb correctness ratio
        List<Double> values = Arrays.asList(0.0, 97.70, 98.71, 99.28, 99.78);// thresholds based on mean and standard deviation
        return findIntervalIndex(subjVerbPercent, values) + 1;// return 1-5 mapped score
    }

    /**
     * traverse the constituency parse tree for a sentence
     *
     * @param child             tree node
     * @param root              root of tree
     * @param sfAttrCountsMap   contituent counts map
     * @param allParentChildren all parent children sequences
     */
    private static void traverseParseTree(Tree child, Tree root, Map<String, Integer> sfAttrCountsMap, Set<String> allParentChildren) {
        if (sfAttrCountsMap.get("FRAGX") == 0) {// if sentence has no FRAG or X tag yet
            if ("FRAG".equals(child.label().value()) || "X".equals(child.label().value())) {// check if this tree has FRAG or X tag
                sfAttrCountsMap.put("FRAGX", 1);// count FRAG or X
                return;
            } else if ("S".equals(child.label().value())) {
                sfAttrCountsMap.put("S", sfAttrCountsMap.get("S") + 1);// count for simple declarative clause
            } else if ("SBAR".equals(child.label().value())) {
                if (!(sbarParents.contains(child.parent(root).value())) || child.getChildrenAsList().stream().noneMatch(node -> sbarChildren.contains(node.label().value()))) {// check for SBAR invalid parents and valid children
                    sfAttrCountsMap.put("SBAR", sfAttrCountsMap.get("SBAR") + 1);// count for invalid SBAR
                }
            }
        }

        extractAllParentChildSeqs(child, allParentChildren);// extract all parent child sequences <Ex:S,NP,VP>

        for (Tree each : child.children()) {
            traverseParseTree(each, root, sfAttrCountsMap, allParentChildren);// recurse through tree
        }

    }

    /**
     * extract all the parent child constituents for the node
     *
     * @param child             tree node
     * @param allParentChildren all parent children sequences
     */
    private static void extractAllParentChildSeqs(Tree child, Set<String> allParentChildren) {
        if (isValidSentFormNode(child)) {// check if node is valid for rules check (that was not covered in FRAG, X, SBAR checks)
            StringBuilder pcr = new StringBuilder();
            List<String> children = child.getChildrenAsList().stream().filter(c -> isValidTag(c.label().value())).map(c -> c.label().value()).collect(Collectors.toList());// get children as list
            if (children.size() > 0) {
                pcr.append(child.label().value().split("-")[0]).append(",");// append the parent
                String s = children.stream().reduce((catStr1, catStr2) -> catStr1.split("-")[0] + "," + catStr2.split("-")[0]).get();// all children as concatenated string, separated by comma
                pcr.append(s);
                allParentChildren.add(pcr.toString());// add to the set
            }
        }
    }

    /**
     * check if constituent tag is ignorable
     *
     * @param tag constituent tag
     * @return boolean
     */
    private static boolean isValidTag(String tag) {
        return !tagsToExclude.contains(tag);
    }

    /**
     * check if the node is valid for rules check
     *
     * @param parent parent node
     * @return boolean
     */
    private static boolean isValidSentFormNode(Tree parent) {
        boolean isValid = false;
        if (!parent.isLeaf() && !Arrays.asList(new String[]{"ROOT", "FRAG", "X", "SBAR"}).contains(parent.label().value())) {// check if parent was not covered in previous rules
            isValid = true;
            for (Tree child : parent.getChildrenAsList()) {// check if child was not covered in previous rules
                if (child.isLeaf() || Arrays.asList(new String[]{"ROOT", "FRAG", "X", "SBAR"}).contains(child.label().value())) {
                    isValid = false;
                }
            }


        }
        return isValid;
    }

    /**
     * part (c iii) - get sentence formation score
     *
     * @param document annotated document
     * @return score
     */
    static int getSentenceFormationScore(Annotation document) {
        int numWrongSents = 0;
        int totalNumSents = 0;
        for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
            totalNumSents++;// count total number of sentences
            boolean fragxPenalty = false;// FRAG/X penalty
            boolean clausePenalty = false;// S penalty
            boolean sbarPenalty = false;// SBAR penalty
            boolean startVerbPenalty = false;// starting with verb penalty
            boolean missingWordsConstPenalty = false;// missing words/constituents penalty
            Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);// get the constituent parse tree
            Map<String, Integer> sfAttrCountsMap = new HashMap<>();// map to count constituents FRAG, S, SBAR
            Set<String> allParentChildren = new HashSet<>();
            sfAttrCountsMap.put("FRAGX", 0);
            sfAttrCountsMap.put("S", 0);
            sfAttrCountsMap.put("SBAR", 0);
            traverseParseTree(tree, tree, sfAttrCountsMap, allParentChildren);// traverse the constituent parse tree
            List<String> posList = sentence.get(CoreAnnotations.TokensAnnotation.class).stream().map(token -> token.get(CoreAnnotations.PartOfSpeechAnnotation.class)).collect(Collectors.toList());// get the pos list

            if (sfAttrCountsMap.get("FRAGX") == 1) {
                fragxPenalty = true;
            } else {
                if (sentStartConflictVerbs.contains(posList.get(0))) {// check if sentence starts with verb
                    startVerbPenalty = true;
                }
                if (sfAttrCountsMap.get("S") == 0) {
                    clausePenalty = true;
                }
                if (sfAttrCountsMap.get("SBAR") > 0) {
                    sbarPenalty = true;
                }
                if (getMissingWordsConstPenalty(allParentChildren) > 0) {
                    missingWordsConstPenalty = true;
                }
            }
            boolean totalPenalty = fragxPenalty || clausePenalty || sbarPenalty || startVerbPenalty || missingWordsConstPenalty;
            if (totalPenalty) {
                numWrongSents++;
            }

        }
        double sentFormMistakes = (numWrongSents / (double) totalNumSents);// ratio of wrong sentences
        List<Double> values = Arrays.asList(0.0, 0.418, 0.485, 0.660, 0.902);// thresholds based on mean and standard deviation
        return 5 - findIntervalIndex(sentFormMistakes, values);// return 1-5 mapped score for correctness
    }

    /**
     * get the number of missing words/constituents by counting rule violations
     *
     * @param allParentChildren all parent children sequences
     * @return count
     */
    private static int getMissingWordsConstPenalty(Set<String> allParentChildren) {
        int missingWordConstCount = 0;
        try {
            BufferedReader rulesReader = Files.newBufferedReader(Paths.get(resPathPrefix + "resources/treebank_rules.txt"));// read trained rules for correct word/constituent sequences
            String nextLine;
            while ((nextLine = rulesReader.readLine()) != null) {
                allTreebankRules.add(nextLine);// add all rules to the set
            }
                /*List<String> mistakesList = allParentChildren.stream().filter(seq -> !allTreebankRules.contains(seq) && grade.equals("high")).collect(Collectors.toList());
                for (String m : mistakesList) {
                    allSeqMistakesFreq.put(m, allSeqMistakesFreq.getOrDefault(m, 0) + 1);
                }*/
            missingWordConstCount = (int) allParentChildren.stream().filter(seq -> !allTreebankRules.contains(seq)).count();// get rule violation counts
        } catch (IOException e) {
            e.printStackTrace();
        }

        return missingWordConstCount;// return count
    }

    /**
     * @param document
     * @param topic
     * @return
     * @throws IOException
     */
    private static Double getTopicRelevanceScore(Annotation document, String topic) throws IOException {
        if (!dictionary.isOpen()) dictionary.open();
        if (topic.contains("\t")) topic = topic.split("\t+")[1];
        Annotation topicAnnotation = processTopic(topic);
        Collection<String> topicNouns = getMainNouns(topicAnnotation, false);
        Collection<String> documentNouns = getMainNouns(document, true);
        Set<LinkedList<ISynset>> topicHyperTrees = topicNouns.stream().flatMap(word -> findRelationalTrees(word, AutograderMain::getHypernyms).stream()).collect(Collectors.toSet());
        Map<String, Double> wordScores = new HashSet<>(documentNouns).stream().collect(Collectors.toMap(Function.identity(), word -> getSimilarityScore(word, topicHyperTrees)));
        dictionary.close();
        if (wordScores.isEmpty()) return 0D;
        double thresholdPercentage = .80D;
        return documentNouns.stream().map(wordScores::get).sorted(Comparator.naturalOrder())
                .skip((int)(documentNouns.size() * (1D - thresholdPercentage)))
                .mapToDouble(Double::doubleValue).average().getAsDouble();
    }

    /**
     * @param word
     * @param topicHyperTrees
     * @return
     */
    private static Double getSimilarityScore(String word, Set<LinkedList<ISynset>> topicHyperTrees) {
        Set<LinkedList<ISynset>> wordHyperTrees = findRelationalTrees(word, AutograderMain::getHypernyms);
        if (!topicHyperTrees.isEmpty() && !wordHyperTrees.isEmpty()) {
            Double score = Sets.cross(wordHyperTrees, topicHyperTrees).stream().map(treePair -> getSimilarityScore(treePair.first, treePair.second)).max(Comparator.naturalOrder()).get();
            return score;
        }
        return 0D;
    }

    /**
     * @param wordTree
     * @param topicTree
     * @return
     */
    private static Double getSimilarityScore(LinkedList<ISynset> wordTree, LinkedList<ISynset> topicTree) {
        Double max_score = 5D;
        if (topicTree.getLast().equals(wordTree.getLast())) return max_score;
        double score = 0D;
        Set<ISynset> wordNetTopic = wordTree.getLast().getRelatedMap().getOrDefault(Pointer.TOPIC, Generics.newArrayList()).stream().map(id -> dictionary.getSynset(id)).collect(Collectors.toSet());
        if (Sets.intersects(wordNetTopic, new HashSet<>(topicTree))) return max_score;

        if (wordTree.contains(topicTree.getLast()) || topicTree.contains(wordTree.getLast())) return max_score;

        Set<ISynset> commonNodes = Sets.intersection(new HashSet<>(wordTree), new HashSet<>(topicTree));
        if (commonNodes.isEmpty()) return score;
        ISynset lowestCommonNode = commonNodes.stream().max(Comparator.comparingInt(wordTree::indexOf)).get();
        int depth1 = wordTree.indexOf(lowestCommonNode) + 1;
        int depth2 = topicTree.indexOf(lowestCommonNode) + 1;
        int depth = depth1 < depth2 ? depth1 : depth2;
        score += max_score * 2 * depth / (wordTree.size() + topicTree.size());

        return score < max_score ? score : max_score;
    }

    /**
     * @param word
     * @param relationFunction
     * @return
     */
    private static Set<LinkedList<ISynset>> findRelationalTrees(String word, Function<ISynset, Set<ISynset>> relationFunction) {
        List<IWordID> wordIDs = dictionary.getIndexWord(word, POS.NOUN).getWordIDs();
        //Conisdering only the first synset.
        wordIDs = Collections.singletonList(wordIDs.iterator().next());
        return wordIDs.stream()
                .map(wordID -> dictionary.getWord(wordID).getSynset())
                .flatMap(synset -> findRelationalTrees(synset, relationFunction).stream())
                .collect(Collectors.toSet());
    }

    /**
     * @param word
     * @param relationFunction
     * @return
     */
    private static Set<LinkedList<ISynset>> findRelationalTrees(ISynset word, Function<ISynset, Set<ISynset>> relationFunction) {
        Set<LinkedList<ISynset>> incompleteHyperTrees = new HashSet<>(Collections.singletonList(new LinkedList<>(Collections.singletonList(word))));
        Set<LinkedList<ISynset>> completeHyperTrees = new HashSet<>();

        while (!incompleteHyperTrees.isEmpty()) {
            LinkedList<ISynset> currentHyperTree = incompleteHyperTrees.iterator().next();
            incompleteHyperTrees.remove(currentHyperTree);
            ISynset head = currentHyperTree.getFirst();
            Set<ISynset> parents = relationFunction.apply(head);
            if (parents.isEmpty()) completeHyperTrees.add(currentHyperTree);
            Stream<LinkedList<ISynset>> updatedHyperTrees = parents.stream().map(parent -> Stream.concat(Stream.of(parent), currentHyperTree.stream()).collect(Collectors.toCollection(LinkedList::new)));
            updatedHyperTrees.forEach(incompleteHyperTrees::add);
        }
        return completeHyperTrees;
    }

    /**
     * @param head
     * @return
     */
    private static Set<ISynset> getHypernyms(ISynset head) {
        return Sets.union(new HashSet<>(head.getRelatedMap().getOrDefault(Pointer.HYPERNYM, new ArrayList<>())), new HashSet<>(head.getRelatedMap().getOrDefault(Pointer.HYPERNYM_INSTANCE, new ArrayList<>()))).stream().map(dictionary::getSynset).collect(Collectors.toSet());
    }

    /**
     * @param head
     * @param relation
     * @return
     */
    private static Set<ISynset> getRelated(ISynset head, Pointer relation) {
        return new HashSet<>(head.getRelatedMap().getOrDefault(relation, new ArrayList<>())).stream().map(dictionary::getSynset).collect(Collectors.toSet());
    }

    /**
     * @param head
     * @return
     */
    private static Set<ISynset> getMeronyms(ISynset head) {
        Set<ISynsetID> partUnion = Sets.union(new HashSet<>(head.getRelatedMap().getOrDefault(Pointer.MERONYM_MEMBER, new ArrayList<>())), new HashSet<>(head.getRelatedMap().getOrDefault(Pointer.MERONYM_PART, new ArrayList<>())));
        return Sets.union(partUnion, new HashSet<>(head.getRelatedMap().getOrDefault(Pointer.MERONYM_SUBSTANCE, new ArrayList<>()))).stream().map(dictionary::getSynset).collect(Collectors.toSet());
    }

    /**
     * @param document
     * @param shouldRepeat
     * @return
     */
    private static Collection<String> getMainNouns(Annotation document, boolean shouldRepeat) {
        List<String> mainNouns = document.get(CoreAnnotations.TokensAnnotation.class).stream()
                .filter(token -> token.get(CoreAnnotations.PartOfSpeechAnnotation.class).contains("NN"))
                .map(token -> token.get(CoreAnnotations.LemmaAnnotation.class))
                .filter(word -> !stopwords_en.contains(word.toLowerCase()) && dictionary.getIndexWord(word, POS.NOUN) != null)
                .collect(Collectors.toList());
        return shouldRepeat ? mainNouns : new HashSet<>(mainNouns);
    }

    /**
     * @param topic
     * @return
     */
    private static Annotation processTopic(String topic) {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        Annotation topicAnnotation = new Annotation(topic);
        pipeline.annotate(topicAnnotation);
        return topicAnnotation;
    }

    /**
     * Essay Autograder: reads essay and grades high/low
     *
     * @param args cmd line args
     */
    public static void main(String[] args) {
        boolean rebuild = false;// flag to rebuild features for training
        if (args.length > 1) {
            rebuild = Boolean.valueOf(args[1]);
        }
        Map<String, Consumer> tasks = new HashMap<>();// tasks map
        tasks.put("test", bool -> testGrader());// test grader
        tasks.put("train", (Consumer<Boolean>) AutograderMain::trainGrader);// train grader
        tasks.get(args[0]).accept(rebuild);
    }

    /**
     * train essay grader
     *
     * @param buildFeatures boolean
     */
    private static void trainGrader(boolean buildFeatures) {
        if (buildFeatures) {// build features
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

                while ((nextRecord = csvReader.readNext()) != null) {// read essays
                    BufferedReader essayReader = Files.newBufferedReader(Paths.get(ioPathPrefix + "input/training/essays/" + nextRecord[0]));
                    StringBuilder essay = new StringBuilder();
                    String line;
                    while ((line = essayReader.readLine()) != null) {
                        essay.append(line).append("\n");
                    }

                    Annotation document = new Annotation(essay.toString());
                    pipeline.annotate(document);
                    int lengthScore = getLengthScore(document);// part (a)
                    int spellScore = spellCheck(document);// part (b)
                    int subjVerbAgrmntScore = getSubjectVerbAgrmntScore(document); // part (c i)
                    int grammarScore = getGrammarScore(document);// part (c ii)
                    int sentFormScore = getSentenceFormationScore(document);// part (c iii)
                    double topicScore = getTopicRelevanceScore(document, nextRecord[1]);// part (d ii)

                    System.out.println(nextRecord[0] + "\t" + lengthScore + "\t" + spellScore + "\t" + subjVerbAgrmntScore + "\t" + grammarScore + "\t" + sentFormScore + "\t" + 0 + "\t" + topicScore + "\t" + nextRecord[2]);

                    csvWriter.writeNext(new String[]{nextRecord[0], String.valueOf(lengthScore), String.valueOf(spellScore), String.valueOf(subjVerbAgrmntScore), String.valueOf(grammarScore), String.valueOf(sentFormScore), String.valueOf(0), String.valueOf(topicScore), nextRecord[2]});// save features to file
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

            Instances trainingDataset = getDataSet(resPathPrefix + "resources/train_features.csv");//load features
            Classifier classifier = new weka.classifiers.functions.SMO();// SMO classifier
            ((SMO) classifier).setOptions(weka.core.Utils.splitOptions("-C 1 -N 2"));// set options C=1, N=2
            classifier.buildClassifier(trainingDataset);
            System.out.println(classifier);

            Evaluation eval = new Evaluation(trainingDataset);// evaluation for cross validation
//            Instances testingDataSet = getDataSet("executable/resources/predict_data_set.csv");
//            eval.evaluateModel(classifier, testingDataSet);
            eval.crossValidateModel(classifier, trainingDataset, 10, new Random(1));// 10-fold cross validation
            System.out.println(eval.toSummaryString());
            Instances predictDataset = getDataSet(resPathPrefix + "resources/predict_data_set.csv");// test predict
            for (Instance i : predictDataset) {
                double value = classifier.classifyInstance(i);
                if (i.classValue() != value) {
                    System.out.println(i);
                    System.out.println(value);
                }

            }

            weka.core.SerializationHelper.write(resPathPrefix + "resources/essay_grader.model", classifier);// save model
        } catch (Exception e) {
            e.printStackTrace();
        }


    }

    /**
     * load the features for training/testing
     *
     * @param filePath path to features file
     * @return Instances
     * @throws IOException file IO exception
     */
    private static Instances getDataSet(String filePath) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filePath));
        Instances dataset = loader.getDataSet();

        dataset.deleteAttributeAt(0);
        dataset.setClassIndex(7);

        return dataset;
    }

    //TODO update function, grade

    /**
     * test grader
     */
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

            while ((nextRecord = csvReader.readNext()) != null) {// read essays
                BufferedReader essayReader = Files.newBufferedReader(Paths.get(ioPathPrefix + "input/testing/essays/" + nextRecord[0]));
                StringBuilder essay = new StringBuilder();
                String line;
                while ((line = essayReader.readLine()) != null) {
                    essay.append(line).append("\n");
                }

                Annotation document = new Annotation(essay.toString());
                pipeline.annotate(document);
                int lengthScore = getLengthScore(document);// part (a)
                int spellScore = spellCheck(document);// part (b)
                int subjVerbAgrmntScore = getSubjectVerbAgrmntScore(document);// part (c i)
                int grammarScore = getGrammarScore(document);// part (c ii)
                int sentFormScore = getSentenceFormationScore(document);// part (c iii)

                double topicScore = getTopicRelevanceScore(document, nextRecord[1]);
                double finalScore = 2.1429 * lengthScore - 0.8571 * spellScore - 0.1429 * subjVerbAgrmntScore * 0.2857 * grammarScore; // final score function
                double intercept = 6.00;// intercept
                String finalGrade = (finalScore + intercept >= 1D) ? "high" : "low";

                System.out.println(nextRecord[0] + ";" + lengthScore + ";" + spellScore + ";" + subjVerbAgrmntScore + ";" + grammarScore + ";" + sentFormScore + ";" + 0 + ";" + topicScore + ";" + (int) finalScore + ";" + finalGrade);
                String scoreDetails = nextRecord[0] + ";" + lengthScore + ";" + spellScore + ";" + subjVerbAgrmntScore + ";" + grammarScore + ";" + sentFormScore + ";" + 0 + ";" + topicScore + ";" + (int) finalScore + ";" + finalGrade + "\n";
                writer.write(scoreDetails);
                essayReader.close();
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


}
