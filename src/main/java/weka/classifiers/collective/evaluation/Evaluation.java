/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    Evaluation.java
 *    Copyright (C) 1999-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.evaluation;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Sourcable;
import weka.classifiers.collective.CollectiveClassifier;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.classifiers.pmml.consumer.PMMLClassifier;
import weka.classifiers.xml.XMLClassifier;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.pmml.PMMLFactory;
import weka.core.pmml.PMMLModel;
import weka.core.xml.KOML;
import weka.core.xml.XMLOptions;
import weka.core.xml.XMLSerialization;

/**
 * Class for evaluating machine learning models.
 * <p/>
 * 
 * -------------------------------------------------------------------
 * <p/>
 * 
 * General options when evaluating a learning scheme from the command-line:
 * <p/>
 * 
 * -t filename <br/>
 * Name of the file with the training data. (required)
 * <p/>
 * 
 * -T filename <br/>
 * Name of the file with the test data. If missing a cross-validation is
 * performed.
 * <p/>
 * 
 * -c index <br/>
 * Index of the class attribute (1, 2, ...; default: last).
 * <p/>
 * 
 * -x number <br/>
 * The number of folds for the cross-validation (default: 10).
 * <p/>
 * 
 * -no-cv <br/>
 * No cross validation. If no test file is provided, no evaluation is done.
 * <p/>
 * 
 * -split-percentage percentage <br/>
 * Sets the percentage for the train/test set split, e.g., 66.
 * <p/>
 * 
 * -preserve-order <br/>
 * Preserves the order in the percentage split instead of randomizing the data
 * first with the seed value ('-s').
 * <p/>
 * 
 * -s seed <br/>
 * Random number seed for the cross-validation and percentage split (default:
 * 1).
 * <p/>
 * 
 * -m filename <br/>
 * The name of a file containing a cost matrix.
 * <p/>
 * 
 * -disable list <br/>
 * A comma separated list of metric names not to include in the output.
 * <p/>
 * 
 * -l filename <br/>
 * Loads classifier from the given file. In case the filename ends with ".xml",
 * a PMML file is loaded or, if that fails, options are loaded from XML.
 * <p/>
 * 
 * -d filename <br/>
 * Saves classifier built from the training data into the given file. In case
 * the filename ends with ".xml" the options are saved XML, not the model.
 * <p/>
 * 
 * -v <br/>
 * Outputs no statistics for the training data.
 * <p/>
 * 
 * -o <br/>
 * Outputs statistics only, not the classifier.
 * <p/>
 * 
 * -i <br/>
 * Outputs information-retrieval statistics per class.
 * <p/>
 * 
 * -k <br/>
 * Outputs information-theoretic statistics.
 * <p/>
 * 
 * -classifications
 * "weka.classifiers.evaluation.output.prediction.AbstractOutput + options" <br/>
 * Uses the specified class for generating the classification output. E.g.:
 * weka.classifiers.evaluation.output.prediction.PlainText or :
 * weka.classifiers.evaluation.output.prediction.CSV
 * 
 * -p range <br/>
 * Outputs predictions for test instances (or the train instances if no test
 * instances provided and -no-cv is used), along with the attributes in the
 * specified range (and nothing else). Use '-p 0' if no attributes are desired.
 * <p/>
 * Deprecated: use "-classifications ..." instead.
 * <p/>
 * 
 * -distribution <br/>
 * Outputs the distribution instead of only the prediction in conjunction with
 * the '-p' option (only nominal classes).
 * <p/>
 * Deprecated: use "-classifications ..." instead.
 * <p/>
 * 
 * -no-predictions <br/>
 * Turns off the collection of predictions in order to conserve memory.
 * <p/>
 * 
 * -r <br/>
 * Outputs cumulative margin distribution (and nothing else).
 * <p/>
 * 
 * -g <br/>
 * Only for classifiers that implement "Graphable." Outputs the graph
 * representation of the classifier (and nothing else).
 * <p/>
 * 
 * -xml filename | xml-string <br/>
 * Retrieves the options from the XML-data instead of the command line.
 * <p/>
 * 
 * -threshold-file file <br/>
 * The file to save the threshold data to. The format is determined by the
 * extensions, e.g., '.arff' for ARFF format or '.csv' for CSV.
 * <p/>
 * 
 * -threshold-label label <br/>
 * The class label to determine the threshold data for (default is the first
 * label)
 * <p/>
 * 
 * -------------------------------------------------------------------
 * <p/>
 * 
 * Example usage as the main of a classifier (called FunkyClassifier):
 * <code> <pre>
 * public static void main(String [] args) {
 *   runClassifier(new FunkyClassifier(), args);
 * }
 * </pre> </code>
 * <p/>
 * 
 * ------------------------------------------------------------------
 * <p/>
 * 
 * Example usage from within an application: <code> <pre>
 * Instances trainInstances = ... instances got from somewhere
 * Instances testInstances = ... instances got from somewhere
 * Classifier scheme = ... scheme got from somewhere
 * 
 * Evaluation evaluation = new Evaluation(trainInstances);
 * evaluation.evaluateModel(scheme, testInstances);
 * System.out.println(evaluation.toSummaryString());
 * </pre> </code>
 * 
 * 
 * @author fracpete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class Evaluation 
  extends weka.classifiers.evaluation.Evaluation {

  /** For serialization */
  private static final long serialVersionUID = -7010314486866816271L;

  /**
   * Initializes all the counters for the evaluation. Use
   * <code>useNoPriors()</code> if the dataset is the test set and you can't
   * initialize with the priors from the training set via
   * <code>setPriors(Instances)</code>.
   * 
   * @param data set of training instances, to get some header information and
   *          prior class distribution information
   * @throws Exception if the class is not defined
   */
  public Evaluation(Instances data) throws Exception {
    super(data);
  }

  /**
   * Initializes all the counters for the evaluation and also takes a cost
   * matrix as parameter. Use <code>useNoPriors()</code> if the dataset is the
   * test set and you can't initialize with the priors from the training set via
   * <code>setPriors(Instances)</code>.
   * 
   * @param data set of training instances, to get some header information and
   *          prior class distribution information
   * @param costMatrix the cost matrix---if null, default costs will be used
   * @throws Exception if cost matrix is not compatible with data, the class is
   *           not defined or the class is numeric
   */
  public Evaluation(Instances data, CostMatrix costMatrix) throws Exception {
    super(data, costMatrix);
  }

  /**
   * Performs a (stratified if class is nominal) cross-validation for a
   * classifier on a set of instances. Now performs a deep copy of the
   * classifier before each call to buildClassifier() (just in case the
   * classifier is not initialized properly).
   * 
   * @param classifier the classifier with any options set.
   * @param data the data on which the cross-validation is to be performed
   * @param numFolds the number of folds for the cross-validation
   * @param random random number generator for randomization
   * @param forPredictionsPrinting varargs parameter that, if supplied, is
   *          expected to hold a
   *          weka.classifiers.evaluation.output.prediction.AbstractOutput
   *          object
   * @throws Exception if a classifier could not be generated successfully or
   *           the class is not defined
   */
  @Override
  public void crossValidateModel(Classifier classifier, Instances data,
      int numFolds, Random random, Object... forPredictionsPrinting)
      throws Exception {

    if (!(classifier instanceof CollectiveClassifier))
      throw new IllegalArgumentException(
	  "Classifier " + classifier.getClass().getName() 
	  + " does not implement " + CollectiveClassifier.class.getName() + "!");
    
    // Make a copy of the data we can reorder
    data = new Instances(data);
    data.randomize(random);
    if (data.classAttribute().isNominal()) {
      data.stratify(numFolds);
    }

    // We assume that the first element is a
    // weka.classifiers.evaluation.output.prediction.AbstractOutput object
    AbstractOutput classificationOutput = null;
    if (forPredictionsPrinting.length > 0) {
      // print the header first
      classificationOutput = (AbstractOutput) forPredictionsPrinting[0];
      classificationOutput.setHeader(data);
      classificationOutput.printHeader();
    }

    // Do the folds
    for (int i = 0; i < numFolds; i++) {
      Instances train = data.trainCV(numFolds, i, random);
      Instances test = data.testCV(numFolds, i);
      setPriors(train);
      Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
      ((CollectiveClassifier) copiedClassifier).buildClassifier(train, test);
      evaluateModel(copiedClassifier, test, forPredictionsPrinting);
    }
    m_NumFolds = numFolds;

    if (classificationOutput != null)
      classificationOutput.printFooter();
  }

  /**
   * Evaluates a classifier with the options given in an array of strings.
   * <p/>
   * 
   * Valid options are:
   * <p/>
   * 
   * -t filename <br/>
   * Name of the file with the training data. (required)
   * <p/>
   * 
   * -T filename <br/>
   * Name of the file with the test data. If missing a cross-validation is
   * performed.
   * <p/>
   * 
   * -c index <br/>
   * Index of the class attribute (1, 2, ...; default: last).
   * <p/>
   * 
   * -x number <br/>
   * The number of folds for the cross-validation (default: 10).
   * <p/>
   * 
   * -no-cv <br/>
   * No cross validation. If no test file is provided, no evaluation is done.
   * <p/>
   * 
   * -split-percentage percentage <br/>
   * Sets the percentage for the train/test set split, e.g., 66.
   * <p/>
   * 
   * -preserve-order <br/>
   * Preserves the order in the percentage split instead of randomizing the data
   * first with the seed value ('-s').
   * <p/>
   * 
   * -s seed <br/>
   * Random number seed for the cross-validation and percentage split (default:
   * 1).
   * <p/>
   * 
   * -m filename <br/>
   * The name of a file containing a cost matrix.
   * <p/>
   * 
   * -l filename <br/>
   * Loads classifier from the given file. In case the filename ends with
   * ".xml",a PMML file is loaded or, if that fails, options are loaded from
   * XML.
   * <p/>
   * 
   * -d filename <br/>
   * Saves classifier built from the training data into the given file. In case
   * the filename ends with ".xml" the options are saved XML, not the model.
   * <p/>
   * 
   * -v <br/>
   * Outputs no statistics for the training data.
   * <p/>
   * 
   * -o <br/>
   * Outputs statistics only, not the classifier.
   * <p/>
   * 
   * -i <br/>
   * Outputs detailed information-retrieval statistics per class.
   * <p/>
   * 
   * -k <br/>
   * Outputs information-theoretic statistics.
   * <p/>
   * 
   * -classifications
   * "weka.classifiers.evaluation.output.prediction.AbstractOutput + options" <br/>
   * Uses the specified class for generating the classification output. E.g.:
   * weka.classifiers.evaluation.output.prediction.PlainText or :
   * weka.classifiers.evaluation.output.prediction.CSV
   * 
   * -p range <br/>
   * Outputs predictions for test instances (or the train instances if no test
   * instances provided and -no-cv is used), along with the attributes in the
   * specified range (and nothing else). Use '-p 0' if no attributes are
   * desired.
   * <p/>
   * Deprecated: use "-classifications ..." instead.
   * <p/>
   * 
   * -distribution <br/>
   * Outputs the distribution instead of only the prediction in conjunction with
   * the '-p' option (only nominal classes).
   * <p/>
   * Deprecated: use "-classifications ..." instead.
   * <p/>
   * 
   * -no-predictions <br/>
   * Turns off the collection of predictions in order to conserve memory.
   * <p/>
   * 
   * -r <br/>
   * Outputs cumulative margin distribution (and nothing else).
   * <p/>
   * 
   * -g <br/>
   * Only for classifiers that implement "Graphable." Outputs the graph
   * representation of the classifier (and nothing else).
   * <p/>
   * 
   * -xml filename | xml-string <br/>
   * Retrieves the options from the XML-data instead of the command line.
   * <p/>
   * 
   * -threshold-file file <br/>
   * The file to save the threshold data to. The format is determined by the
   * extensions, e.g., '.arff' for ARFF format or '.csv' for CSV.
   * <p/>
   * 
   * -threshold-label label <br/>
   * The class label to determine the threshold data for (default is the first
   * label)
   * <p/>
   * 
   * @param classifierString class of machine learning classifier as a string
   * @param options the array of string containing the options
   * @throws Exception if model could not be evaluated successfully
   * @return a string describing the results
   */
  public static String evaluateModel(String classifierString, String[] options)
      throws Exception {

    Classifier classifier;

    // Create classifier
    try {
      classifier =
      // (Classifier)Class.forName(classifierString).newInstance();
      AbstractClassifier.forName(classifierString, null);
    } catch (Exception e) {
      throw new Exception("Can't find class with name " + classifierString
          + '.');
    }
    return evaluateModel(classifier, options);
  }

  /**
   * Evaluates a classifier with the options given in an array of strings.
   * <p/>
   * 
   * Valid options are:
   * <p/>
   * 
   * -t name of training file <br/>
   * Name of the file with the training data. (required)
   * <p/>
   * 
   * -T name of test file <br/>
   * Name of the file with the test data. If missing a cross-validation is
   * performed.
   * <p/>
   * 
   * -c class index <br/>
   * Index of the class attribute (1, 2, ...; default: last).
   * <p/>
   * 
   * -x number of folds <br/>
   * The number of folds for the cross-validation (default: 10).
   * <p/>
   * 
   * -no-cv <br/>
   * No cross validation. If no test file is provided, no evaluation is done.
   * <p/>
   * 
   * -split-percentage percentage <br/>
   * Sets the percentage for the train/test set split, e.g., 66.
   * <p/>
   * 
   * -preserve-order <br/>
   * Preserves the order in the percentage split instead of randomizing the data
   * first with the seed value ('-s').
   * <p/>
   * 
   * -s seed <br/>
   * Random number seed for the cross-validation and percentage split (default:
   * 1).
   * <p/>
   * 
   * -m file with cost matrix <br/>
   * The name of a file containing a cost matrix.
   * <p/>
   * 
   * -l filename <br/>
   * Loads classifier from the given file. In case the filename ends with
   * ".xml",a PMML file is loaded or, if that fails, options are loaded from
   * XML.
   * <p/>
   * 
   * -d filename <br/>
   * Saves classifier built from the training data into the given file. In case
   * the filename ends with ".xml" the options are saved XML, not the model.
   * <p/>
   * 
   * -v <br/>
   * Outputs no statistics for the training data.
   * <p/>
   * 
   * -o <br/>
   * Outputs statistics only, not the classifier.
   * <p/>
   * 
   * -i <br/>
   * Outputs detailed information-retrieval statistics per class.
   * <p/>
   * 
   * -k <br/>
   * Outputs information-theoretic statistics.
   * <p/>
   * 
   * -classifications
   * "weka.classifiers.evaluation.output.prediction.AbstractOutput + options" <br/>
   * Uses the specified class for generating the classification output. E.g.:
   * weka.classifiers.evaluation.output.prediction.PlainText or :
   * weka.classifiers.evaluation.output.prediction.CSV
   * 
   * -p range <br/>
   * Outputs predictions for test instances (or the train instances if no test
   * instances provided and -no-cv is used), along with the attributes in the
   * specified range (and nothing else). Use '-p 0' if no attributes are
   * desired.
   * <p/>
   * Deprecated: use "-classifications ..." instead.
   * <p/>
   * 
   * -distribution <br/>
   * Outputs the distribution instead of only the prediction in conjunction with
   * the '-p' option (only nominal classes).
   * <p/>
   * Deprecated: use "-classifications ..." instead.
   * <p/>
   * 
   * -no-predictions <br/>
   * Turns off the collection of predictions in order to conserve memory.
   * <p/>
   * 
   * -r <br/>
   * Outputs cumulative margin distribution (and nothing else).
   * <p/>
   * 
   * -g <br/>
   * Only for classifiers that implement "Graphable." Outputs the graph
   * representation of the classifier (and nothing else).
   * <p/>
   * 
   * -xml filename | xml-string <br/>
   * Retrieves the options from the XML-data instead of the command line.
   * <p/>
   * 
   * @param classifier machine learning classifier
   * @param options the array of string containing the options
   * @throws Exception if model could not be evaluated successfully
   * @return a string describing the results
   */
  public static String evaluateModel(Classifier classifier, String[] options)
      throws Exception {

    Instances train = null, tempTrain, test = null, template = null;
    int seed = 1, folds = 10, classIndex = -1;
    boolean noCrossValidation = false;
    String trainFileName, testFileName, sourceClass, classIndexString, seedString, foldsString, objectInputFileName, objectOutputFileName;
    boolean noOutput = false, trainStatistics = true, printMargins = false, printComplexityStatistics = false, printGraph = false, classStatistics = false, printSource = false;
    StringBuffer text = new StringBuffer();
    DataSource trainSource = null, testSource = null;
    ObjectInputStream objectInputStream = null;
    BufferedInputStream xmlInputStream = null;
    CostMatrix costMatrix = null;
    StringBuffer schemeOptionsText = null;
    long trainTimeStart = 0, trainTimeElapsed = 0, testTimeStart = 0, testTimeElapsed = 0;
    String xml = "";
    String[] optionsTmp = null;
    Classifier classifierBackup;
    int actualClassIndex = -1; // 0-based class index
    String splitPercentageString = "";
    double splitPercentage = -1;
    boolean preserveOrder = false;
    boolean trainSetPresent = false;
    boolean testSetPresent = false;
    boolean discardPredictions = false;
    String thresholdFile;
    String thresholdLabel;
    StringBuffer predsBuff = null; // predictions from cross-validation
    AbstractOutput classificationOutput = null;

    // help requested?
    if (Utils.getFlag("h", options) || Utils.getFlag("help", options)) {

      // global info requested as well?
      boolean globalInfo = Utils.getFlag("synopsis", options)
          || Utils.getFlag("info", options);

      throw new Exception("\nHelp requested."
          + makeOptionString(classifier, globalInfo));
    }

    String metricsToDisable = Utils.getOption("disable", options);
    List<String> disableList = new ArrayList<String>();
    if (metricsToDisable.length() > 0) {
      String[] parts = metricsToDisable.split(",");
      for (String p : parts) {
        disableList.add(p.trim().toLowerCase());
      }
    }

    try {
      // do we get the input from XML instead of normal parameters?
      xml = Utils.getOption("xml", options);
      if (!xml.equals(""))
        options = new XMLOptions(xml).toArray();

      // is the input model only the XML-Options, i.e. w/o built model?
      optionsTmp = new String[options.length];
      for (int i = 0; i < options.length; i++)
        optionsTmp[i] = options[i];

      String tmpO = Utils.getOption('l', optionsTmp);
      // if (Utils.getOption('l', optionsTmp).toLowerCase().endsWith(".xml")) {
      if (tmpO.endsWith(".xml")) {
        // try to load file as PMML first
        boolean success = false;
        try {
          PMMLModel pmmlModel = PMMLFactory.getPMMLModel(tmpO);
          if (pmmlModel instanceof PMMLClassifier) {
            classifier = ((PMMLClassifier) pmmlModel);
            success = true;
          }
        } catch (IllegalArgumentException ex) {
          success = false;
        }
        if (!success) {
          // load options from serialized data ('-l' is automatically erased!)
          XMLClassifier xmlserial = new XMLClassifier();
          OptionHandler cl = (OptionHandler) xmlserial.read(Utils.getOption(
              'l', options));

          // merge options
          optionsTmp = new String[options.length + cl.getOptions().length];
          System.arraycopy(cl.getOptions(), 0, optionsTmp, 0,
              cl.getOptions().length);
          System.arraycopy(options, 0, optionsTmp, cl.getOptions().length,
              options.length);
          options = optionsTmp;
        }
      }

      noCrossValidation = Utils.getFlag("no-cv", options);
      // Get basic options (options the same for all schemes)
      classIndexString = Utils.getOption('c', options);
      if (classIndexString.length() != 0) {
        if (classIndexString.equals("first"))
          classIndex = 1;
        else if (classIndexString.equals("last"))
          classIndex = -1;
        else
          classIndex = Integer.parseInt(classIndexString);
      }
      trainFileName = Utils.getOption('t', options);
      objectInputFileName = Utils.getOption('l', options);
      objectOutputFileName = Utils.getOption('d', options);
      testFileName = Utils.getOption('T', options);
      foldsString = Utils.getOption('x', options);
      if (foldsString.length() != 0) {
        folds = Integer.parseInt(foldsString);
      }
      seedString = Utils.getOption('s', options);
      if (seedString.length() != 0) {
        seed = Integer.parseInt(seedString);
      }
      if (trainFileName.length() == 0) {
        if (objectInputFileName.length() == 0) {
          throw new Exception(
              "No training file and no object input file given.");
        }
        if (testFileName.length() == 0) {
          throw new Exception("No training file and no test file given.");
        }
      } else if ((objectInputFileName.length() != 0)
          && ((testFileName.length() == 0))) {
        throw new Exception("No test file provided: can't " + "use both train and model file.");
      }
      try {
        if (trainFileName.length() != 0) {
          trainSetPresent = true;
          trainSource = new DataSource(trainFileName);
        }
        if (testFileName.length() != 0) {
          testSetPresent = true;
          testSource = new DataSource(testFileName);
        }
        if (objectInputFileName.length() != 0) {
          if (objectInputFileName.endsWith(".xml")) {
            // if this is the case then it means that a PMML classifier was
            // successfully loaded earlier in the code
            objectInputStream = null;
            xmlInputStream = null;
          } else {
            InputStream is = new FileInputStream(objectInputFileName);
            if (objectInputFileName.endsWith(".gz")) {
              is = new GZIPInputStream(is);
            }
            // load from KOML?
            if (!(objectInputFileName.endsWith(".koml") && KOML.isPresent())) {
              objectInputStream = new ObjectInputStream(is);
              xmlInputStream = null;
            } else {
              objectInputStream = null;
              xmlInputStream = new BufferedInputStream(is);
            }
          }
        }
      } catch (Exception e) {
        throw new Exception("Can't open file " + e.getMessage() + '.');
      }
      if (testSetPresent) {
        template = test = testSource.getStructure();
        if (classIndex != -1) {
          test.setClassIndex(classIndex - 1);
        } else {
          if ((test.classIndex() == -1) || (classIndexString.length() != 0))
            test.setClassIndex(test.numAttributes() - 1);
        }
        actualClassIndex = test.classIndex();
      } else {
        // percentage split
        splitPercentageString = Utils.getOption("split-percentage", options);
        if (splitPercentageString.length() != 0) {
          if (foldsString.length() != 0)
            throw new Exception(
                "Percentage split cannot be used in conjunction with "
                    + "cross-validation ('-x').");
          splitPercentage = Double.parseDouble(splitPercentageString);
          if ((splitPercentage <= 0) || (splitPercentage >= 100))
            throw new Exception("Percentage split value needs be >0 and <100.");
        } else {
          splitPercentage = -1;
        }
        preserveOrder = Utils.getFlag("preserve-order", options);
        if (preserveOrder) {
          if (splitPercentage == -1)
            throw new Exception(
                "Percentage split ('-percentage-split') is missing.");
        }
        // create new train/test sources
        if (splitPercentage > 0) {
          testSetPresent = true;
          Instances tmpInst = trainSource.getDataSet(actualClassIndex);
          if (!preserveOrder)
            tmpInst.randomize(new Random(seed));
          int trainSize = (int) Math.round(tmpInst.numInstances()
              * splitPercentage / 100);
          int testSize = tmpInst.numInstances() - trainSize;
          Instances trainInst = new Instances(tmpInst, 0, trainSize);
          Instances testInst = new Instances(tmpInst, trainSize, testSize);
          trainSource = new DataSource(trainInst);
          testSource = new DataSource(testInst);
          template = test = testSource.getStructure();
          if (classIndex != -1) {
            test.setClassIndex(classIndex - 1);
          } else {
            if ((test.classIndex() == -1) || (classIndexString.length() != 0))
              test.setClassIndex(test.numAttributes() - 1);
          }
          actualClassIndex = test.classIndex();
        }
      }
      if (trainSetPresent) {
        template = train = trainSource.getStructure();
        if (classIndex != -1) {
          train.setClassIndex(classIndex - 1);
        } else {
          if ((train.classIndex() == -1) || (classIndexString.length() != 0))
            train.setClassIndex(train.numAttributes() - 1);
        }
        actualClassIndex = train.classIndex();
        if (!(classifier instanceof weka.classifiers.misc.InputMappedClassifier)) {
          if ((testSetPresent) && !test.equalHeaders(train)) {
            throw new IllegalArgumentException(
                "Train and test file not compatible!\n"
                    + test.equalHeadersMsg(train));
          }
        }
      }
      if (template == null) {
        throw new Exception("No actual dataset provided to use as template");
      }
      costMatrix = handleCostOption(Utils.getOption('m', options),
          template.numClasses());

      classStatistics = Utils.getFlag('i', options);
      noOutput = Utils.getFlag('o', options);
      trainStatistics = !Utils.getFlag('v', options);
      printComplexityStatistics = Utils.getFlag('k', options);
      printMargins = Utils.getFlag('r', options);
      printGraph = Utils.getFlag('g', options);
      sourceClass = Utils.getOption('z', options);
      printSource = (sourceClass.length() != 0);
      thresholdFile = Utils.getOption("threshold-file", options);
      thresholdLabel = Utils.getOption("threshold-label", options);

      String classifications = Utils.getOption("classifications", options);
      String classificationsOld = Utils.getOption("p", options);
      if (classifications.length() > 0) {
        noOutput = true;
        classificationOutput = AbstractOutput.fromCommandline(classifications);
        if (classificationOutput == null)
          throw new Exception(
              "Failed to instantiate class for classification output: "
                  + classifications);
        classificationOutput.setHeader(template);
      }
      // backwards compatible with old "-p range" and "-distribution" options
      else if (classificationsOld.length() > 0) {
        noOutput = true;
        classificationOutput = new PlainText();
        classificationOutput.setHeader(template);
        if (!classificationsOld.equals("0"))
          classificationOutput.setAttributes(classificationsOld);
        classificationOutput.setOutputDistribution(Utils.getFlag(
            "distribution", options));
      }
      // -distribution flag needs -p option
      else {
        if (Utils.getFlag("distribution", options))
          throw new Exception("Cannot print distribution without '-p' option!");
      }
      discardPredictions = Utils.getFlag("no-predictions", options);
      if (discardPredictions && (classificationOutput != null))
        throw new Exception(
            "Cannot discard predictions ('-no-predictions') and output predictions at the same time ('-classifications/-p')!");

      // if no training file given, we don't have any priors
      if ((!trainSetPresent) && (printComplexityStatistics))
        throw new Exception(
            "Cannot print complexity statistics ('-k') without training file ('-t')!");

      // If a model file is given, we can't process
      // scheme-specific options
      if (objectInputFileName.length() != 0) {
        Utils.checkForRemainingOptions(options);
      } else {

        // Set options for classifier
        if (classifier instanceof OptionHandler) {
          for (int i = 0; i < options.length; i++) {
            if (options[i].length() != 0) {
              if (schemeOptionsText == null) {
                schemeOptionsText = new StringBuffer();
              }
              if (options[i].indexOf(' ') != -1) {
                schemeOptionsText.append('"' + options[i] + "\" ");
              } else {
                schemeOptionsText.append(options[i] + " ");
              }
            }
          }
          ((OptionHandler) classifier).setOptions(options);
        }
      }

      Utils.checkForRemainingOptions(options);
    } catch (Exception e) {
      throw new Exception("\nWeka exception: " + e.getMessage()
          + makeOptionString(classifier, false));
    }

    if (objectInputFileName.length() != 0) {
      // Load classifier from file
      if (objectInputStream != null) {
        classifier = (Classifier) objectInputStream.readObject();
        // try and read a header (if present)
        Instances savedStructure = null;
        try {
          savedStructure = (Instances) objectInputStream.readObject();
        } catch (Exception ex) {
          // don't make a fuss
        }
        if (savedStructure != null) {
          // test for compatibility with template
          if (!template.equalHeaders(savedStructure)) {
            throw new Exception("training and test set are not compatible\n"
                + template.equalHeadersMsg(savedStructure));
          }
        }
        objectInputStream.close();
      } else if (xmlInputStream != null) {
        // whether KOML is available has already been checked (objectInputStream
        // would null otherwise)!
        classifier = (Classifier) KOML.read(xmlInputStream);
        xmlInputStream.close();
      }
    }

    // Setup up evaluation objects
    Evaluation trainingEvaluation = new Evaluation(new Instances(template, 0),
        costMatrix);
    Evaluation testingEvaluation = new Evaluation(new Instances(template, 0),
        costMatrix);
    if (classifier instanceof weka.classifiers.misc.InputMappedClassifier) {
      Instances mappedClassifierHeader = ((weka.classifiers.misc.InputMappedClassifier) classifier)
          .getModelHeader(new Instances(template, 0));

      trainingEvaluation = new Evaluation(new Instances(mappedClassifierHeader,
          0), costMatrix);
      testingEvaluation = new Evaluation(new Instances(mappedClassifierHeader,
          0), costMatrix);
    }
    trainingEvaluation.setDiscardPredictions(discardPredictions);
    trainingEvaluation.dontDisplayMetrics(disableList);
    testingEvaluation.setDiscardPredictions(discardPredictions);
    testingEvaluation.dontDisplayMetrics(disableList);

    // disable use of priors if no training file given
    if (!trainSetPresent)
      testingEvaluation.useNoPriors();

    // backup of fully setup classifier for cross-validation
    classifierBackup = AbstractClassifier.makeCopy(classifier);

    // Build the classifier if no object file provided
    if (objectInputFileName.length() == 0) {
      // Build classifier in one go
      tempTrain = trainSource.getDataSet(actualClassIndex);

      if (classifier instanceof weka.classifiers.misc.InputMappedClassifier
          && !trainingEvaluation.getHeader().equalHeaders(tempTrain)) {
        // we need to make a new dataset that maps the training instances to
        // the structure expected by the mapped classifier - this is only
        // to ensure that the structure and priors computed by the *testing*
        // evaluation object is correct with respect to the mapped classifier
        Instances mappedClassifierDataset = ((weka.classifiers.misc.InputMappedClassifier) classifier)
            .getModelHeader(new Instances(template, 0));
        for (int zz = 0; zz < tempTrain.numInstances(); zz++) {
          Instance mapped = ((weka.classifiers.misc.InputMappedClassifier) classifier)
              .constructMappedInstance(tempTrain.instance(zz));
          mappedClassifierDataset.add(mapped);
        }
        tempTrain = mappedClassifierDataset;
      }

      trainingEvaluation.setPriors(tempTrain);
      testingEvaluation.setPriors(tempTrain);
      trainTimeStart = System.currentTimeMillis();
      classifier.buildClassifier(tempTrain);
      trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
    }

    // backup of fully trained classifier for printing the classifications
    if (classificationOutput != null) {
      if (classifier instanceof weka.classifiers.misc.InputMappedClassifier) {
        classificationOutput.setHeader(trainingEvaluation.getHeader());
      }
    }

    // Save the classifier if an object output file is provided
    if (objectOutputFileName.length() != 0) {
      OutputStream os = new FileOutputStream(objectOutputFileName);
      // binary
      if (!(objectOutputFileName.endsWith(".xml") || (objectOutputFileName
          .endsWith(".koml") && KOML.isPresent()))) {
        if (objectOutputFileName.endsWith(".gz")) {
          os = new GZIPOutputStream(os);
        }
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(os);
        objectOutputStream.writeObject(classifier);
        if (template != null) {
          objectOutputStream.writeObject(template);
        }
        objectOutputStream.flush();
        objectOutputStream.close();
      }
      // KOML/XML
      else {
        BufferedOutputStream xmlOutputStream = new BufferedOutputStream(os);
        if (objectOutputFileName.endsWith(".xml")) {
          XMLSerialization xmlSerial = new XMLClassifier();
          xmlSerial.write(xmlOutputStream, classifier);
        } else
        // whether KOML is present has already been checked
        // if not present -> ".koml" is interpreted as binary - see above
        if (objectOutputFileName.endsWith(".koml")) {
          KOML.write(xmlOutputStream, classifier);
        }
        xmlOutputStream.close();
      }
    }

    // If classifier is drawable output string describing graph
    if ((classifier instanceof Drawable) && (printGraph)) {
      return ((Drawable) classifier).graph();
    }

    // Output the classifier as equivalent source
    if ((classifier instanceof Sourcable) && (printSource)) {
      return wekaStaticWrapper((Sourcable) classifier, sourceClass);
    }

    // Output model
    if (!(noOutput || printMargins)) {
      if (classifier instanceof OptionHandler) {
        if (schemeOptionsText != null) {
          text.append("\nOptions: " + schemeOptionsText);
          text.append("\n");
        }
      }
      text.append("\n" + classifier.toString() + "\n");
    }

    if (!printMargins && (costMatrix != null)) {
      text.append("\n=== Evaluation Cost Matrix ===\n\n");
      text.append(costMatrix.toString());
    }

    // Output test instance predictions only
    if (classificationOutput != null) {
      DataSource source = testSource;
      predsBuff = new StringBuffer();
      classificationOutput.setBuffer(predsBuff);
      // no test set -> use train set
      if (source == null && noCrossValidation) {
        source = trainSource;
        predsBuff.append("\n=== Predictions on training data ===\n\n");
      } else {
        predsBuff.append("\n=== Predictions on test data ===\n\n");
      }
      if (source != null)
        classificationOutput.print(classifier, source);
    }

    // Compute error estimate from training data
    if ((trainStatistics) && (trainSetPresent)) {

      testTimeStart = System.currentTimeMillis();
      trainingEvaluation.evaluateModel(classifier,
	  trainSource.getDataSet(actualClassIndex));
      testTimeElapsed = System.currentTimeMillis() - testTimeStart;

      // Print the results of the training evaluation
      if (printMargins) {
        return trainingEvaluation.toCumulativeMarginDistributionString();
      } else {
        if (classificationOutput == null) {
          text.append("\nTime taken to build model: "
              + Utils.doubleToString(trainTimeElapsed / 1000.0, 2) + " seconds");

          if (splitPercentage > 0)
            text.append("\nTime taken to test model on training split: ");
          else
            text.append("\nTime taken to test model on training data: ");
          text.append(Utils.doubleToString(testTimeElapsed / 1000.0, 2)
              + " seconds");

          if (splitPercentage > 0)
            text.append(trainingEvaluation.toSummaryString(
                "\n\n=== Error on training" + " split ===\n",
                printComplexityStatistics));
          else
            text.append(trainingEvaluation.toSummaryString(
                "\n\n=== Error on training" + " data ===\n",
                printComplexityStatistics));

          if (template.classAttribute().isNominal()) {
            if (classStatistics) {
              text.append("\n\n" + trainingEvaluation.toClassDetailsString());
            }
            if (!noCrossValidation)
              text.append("\n\n" + trainingEvaluation.toMatrixString());
          }
        }
      }
    }

    // Compute proper error estimates
    if (testSource != null) {
      // Testing is on the supplied test data
      testSource.reset();
      test = testSource.getStructure(test.classIndex());
      Instance testInst;
      while (testSource.hasMoreElements(test)) {
        testInst = testSource.nextElement(test);
        testingEvaluation.evaluateModelOnceAndRecordPrediction(classifier,
            testInst);
      }

      if (splitPercentage > 0) {
        if (classificationOutput == null) {
          text.append("\n\n"
              + testingEvaluation.toSummaryString(
                  "=== Error on test split ===\n", printComplexityStatistics));
        }
      } else {
        if (classificationOutput == null) {
          text.append("\n\n"
              + testingEvaluation.toSummaryString(
                  "=== Error on test data ===\n", printComplexityStatistics));
        }
      }

    } else if (trainSource != null) {
      if (!noCrossValidation) {
        // Testing is via cross-validation on training data
        Random random = new Random(seed);
        // use untrained (!) classifier for cross-validation
        classifier = AbstractClassifier.makeCopy(classifierBackup);
        if (classificationOutput == null) {
          testingEvaluation.crossValidateModel(classifier,
              trainSource.getDataSet(actualClassIndex), folds, random);
          if (template.classAttribute().isNumeric()) {
            text.append("\n\n\n"
                + testingEvaluation.toSummaryString(
                    "=== Cross-validation ===\n", printComplexityStatistics));
          } else {
            text.append("\n\n\n"
                + testingEvaluation.toSummaryString("=== Stratified "
                    + "cross-validation ===\n", printComplexityStatistics));
          }
        } else {
          predsBuff = new StringBuffer();
          classificationOutput.setBuffer(predsBuff);
          predsBuff.append("\n=== Predictions under cross-validation ===\n\n");
          testingEvaluation.crossValidateModel(classifier,
              trainSource.getDataSet(actualClassIndex), folds, random,
              classificationOutput);
        }
      }
    }
    if (template.classAttribute().isNominal()) {
      if (classStatistics && !noCrossValidation
          && (classificationOutput == null)) {
        text.append("\n\n" + testingEvaluation.toClassDetailsString());
      }
      if (!noCrossValidation && (classificationOutput == null))
        text.append("\n\n" + testingEvaluation.toMatrixString());

    }

    // predictions from cross-validation?
    if (predsBuff != null) {
      text.append("\n" + predsBuff);
    }

    if ((thresholdFile.length() != 0) && template.classAttribute().isNominal()) {
      int labelIndex = 0;
      if (thresholdLabel.length() != 0)
        labelIndex = template.classAttribute().indexOfValue(thresholdLabel);
      if (labelIndex == -1)
        throw new IllegalArgumentException("Class label '" + thresholdLabel
            + "' is unknown!");
      ThresholdCurve tc = new ThresholdCurve();
      Instances result = tc.getCurve(testingEvaluation.predictions(),
          labelIndex);
      DataSink.write(thresholdFile, result);
    }

    return text.toString();
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 2019 $");
  }
}
