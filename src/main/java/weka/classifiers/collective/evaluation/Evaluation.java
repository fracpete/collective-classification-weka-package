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

import java.util.Enumeration;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.collective.CollectiveClassifier;
import weka.classifiers.collective.util.CollectiveHelper;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SerializationHelper;
import weka.core.SingleIndex;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Class for evaluating machine learning models.
 * <p/>
 * In case of cross-validation, train and test set are swapped, as there
 * is usually more unlabeled than labeled data. For instance, with 10-fold
 * CV you get 10% labeled and 90% unlabeled data.
 * <p/>
 * Use "-h" or "-help" on the command-line of a collective classifier to see
 * all the available options, e.g.:
 * <pre>
 * java [classpath] weka.classifiers.collective.meta.YATSI -help
 * </pre>
 * 
 * @author fracpete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class Evaluation 
  extends weka.classifiers.evaluation.Evaluation {

  /** For serialization */
  private static final long serialVersionUID = -7010314486866816271L;

  /** whether to swap train/test folds in case of cross-validation. */
  protected boolean m_SwapFolds;
  
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
   * Sets whether to swap the train/test folds in case of cross-validation.
   * Basically inverts training/test size.
   * 
   * @param value	true if to swap
   */
  public void setSwapFolds(boolean value) {
    m_SwapFolds = value;
  }
  
  /**
   * Returns whether the train/test folds are swapped in case of 
   * cross-validation.
   * 
   * @return		true if swapped
   */
  public boolean getSwapFolds() {
    return m_SwapFolds;
  }
  
  /**
   * Performs a (stratified if class is nominal) cross-validation for a
   * classifier on a set of instances. Now performs a deep copy of the
   * classifier before each call to buildClassifier() (just in case the
   * classifier is not initialized properly).
   * Train and test set are swapped, as there is usually more unlabeled than 
   * labeled data. For instance, with 10-fold CV you get 10% labeled and 
   * 90% unlabeled data.
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
      Instances test;
      Instances train;
      if (m_SwapFolds) {
	test = data.trainCV(numFolds, i, random);
	train = data.testCV(numFolds, i);
      }
      else {
	train = data.trainCV(numFolds, i, random);
	test = data.testCV(numFolds, i);
      }
      setPriors(train);
      Instances unlabeled = CollectiveHelper.removeLabels(test, true);
      Classifier copiedClassifier = AbstractClassifier.makeCopy(classifier);
      if (copiedClassifier instanceof CollectiveClassifier)
	((CollectiveClassifier) copiedClassifier).buildClassifier(train, unlabeled);
      else
	copiedClassifier.buildClassifier(train);
      evaluateModel(copiedClassifier, test, forPredictionsPrinting);
    }
    m_NumFolds = numFolds;

    if (classificationOutput != null)
      classificationOutput.printFooter();
  }

  /**
   * Evaluates a classifier with the options given in an array of strings.
   * 
   * @param classifierString class of machine learning classifier as a string
   * @param options the array of string containing the options
   * @throws Exception if model could not be evaluated successfully
   * @return a string describing the results
   */
  public static String evaluateModel(String classifierString, String[] options) throws Exception {
    Classifier classifier;

    try {
      classifier = AbstractClassifier.forName(classifierString, null);
    } 
    catch (Exception e) {
      throw new Exception("Can't find class with name " + classifierString + '.');
    }

    return evaluateModel(classifier, options);
  }

  /**
   * Generates a help string.
   * 
   * @param classifier	the classifier to generate the string for
   * @return		the generated string
   */
  protected static String generateHelp(CollectiveClassifier classifier) {
    StringBuilder	result;
    Enumeration 	enm;
    Option		option;
    
    result = new StringBuilder();
    result.append("\n");
    result.append("Help\n");
    result.append("====\n");
    result.append("\n");
    result.append("General options:\n");
    result.append("\n");
    result.append("-t <file>\n");
    result.append("\tTraining set\n");
    result.append("\n");
    result.append("-T <file>\n");
    result.append("\tTest set\n");
    result.append("\n");
    result.append("-c <index>\n");
    result.append("\tClass index (1-based or 'first' or 'last')\n");
    result.append("\tdefault: last\n");
    result.append("\n");
    result.append("-l <file>\n");
    result.append("\tSerialized model to use, requires '-T' but does not allow '-t'\n");
    result.append("\n");
    result.append("-d <file>\n");
    result.append("\tSerializes a built model, not available when using cross-validation\n");
    result.append("\n");
    result.append("-x <folds>\n");
    result.append("\tThe number of folds for cross-validation (if '-T' is omitted)\n");
    result.append("\tUse -1 for leave-one-out cross-validation\n");
    result.append("\tdefault: 10\n");
    result.append("\n");
    result.append("-swap-folds\n");
    result.append("\tSwaps train and test folds in case of cross-validation\n");
    result.append("\n");
    result.append("-s <number>\n");
    result.append("\tThe seed value for randomization\n");
    result.append("\tdefault: 1\n");
    result.append("\n");
    result.append("-split-percentage <0-100>\n");
    result.append("\tSplits the training set into train and test set with the\n");
    result.append("\tspecified amount set aside for training\n");
    result.append("\n");
    result.append("-preserve-order\n");
    result.append("\tTurns off randomization when using '-split-percentage'\n");
    result.append("\n");
    result.append("Classifier options:\n");
    result.append("\n");

    // Get scheme-specific options
    if (classifier instanceof OptionHandler) {
      result.append("Options specific to " + classifier.getClass().getName() + ":\n");
      result.append("\n");
      enm = ((OptionHandler) classifier).listOptions();
      while (enm.hasMoreElements()) {
        option = (Option) enm.nextElement();
        result.append(option.synopsis() + "\n");
        result.append(option.description() + "\n");
      }
    }
    
    return result.toString();
  }
  
  /**
   * Evaluates a classifier with the options given in an array of strings.
   * 
   * @param classifier machine learning classifier
   * @param options the array of string containing the options
   * @throws Exception if model could not be evaluated successfully
   * @return a string describing the results
   */
  public static String evaluateModel(CollectiveClassifier classifier, String[] options) throws Exception {
    String			tmpStr;
    String			trainFile;
    String			testFile;
    String			modelFile;
    String			dumpFile;
    int				folds;
    int				seed;
    double			percentage;
    boolean			preserve;
    int 			trainSize;
    int 			testSize;
    SingleIndex			classIndex;
    Instances			train;
    Instances			test;
    CollectiveClassifier	model;
    Object[]			obj;
    Evaluation			eval;
    boolean			swap;

    // help?
    if (Utils.getFlag('h', options) || Utils.getFlag("help", options))
      return generateHelp(classifier);
    
    // class index
    tmpStr = Utils.getOption('c', options);
    if (tmpStr.length() == 0)
      classIndex = new SingleIndex("last");
    else
      classIndex = new SingleIndex(tmpStr);
    
    // seed
    tmpStr = Utils.getOption('s', options);
    if (tmpStr.length() == 0)
      tmpStr = "1";
    try {
      seed = Integer.parseInt(tmpStr);
    }
    catch (Exception e) {
      throw new IllegalArgumentException("Failed to parse seed ('-s') string '" + tmpStr + "'!", e);
    }
    
    // training
    trainFile = Utils.getOption('t', options);
    train     = null;
    if (trainFile.length() > 0) {
      train = DataSource.read(trainFile);
      if (train == null)
	throw new IllegalArgumentException("Failed to read training set ('-t'): " + trainFile);
      classIndex.setUpper(train.numAttributes() - 1);
      train.setClassIndex(classIndex.getIndex());
    }
    
    // testing
    testFile = Utils.getOption('T', options);
    test     = null;
    if (testFile.length() > 0) {
      test = DataSource.read(testFile);
      if (test == null)
	throw new IllegalArgumentException("Failed to read test set ('-T'): " + testFile);
      classIndex.setUpper(test.numAttributes() - 1);
      test.setClassIndex(classIndex.getIndex());
    }

    // split?
    if (test == null) {
      tmpStr = Utils.getOption("split-percentage", options);
      if (tmpStr.length() > 0) {
	try {
	  percentage = Double.parseDouble(tmpStr);
	}
	catch (Exception e) {
	  throw new IllegalArgumentException("Failed to parse split-percentage: " + tmpStr, e);
	}
	preserve = Utils.getFlag("preserve-order", options);
        if (!preserve)
          train.randomize(new Random(seed));
        trainSize = (int) Math.round(train.numInstances() * percentage / 100);
        testSize  = train.numInstances() - trainSize;
        test      = new Instances(train, trainSize, testSize);
        train     = new Instances(train, 0, trainSize);
        folds     = 0;
      }
    }
    
    // serialized model?
    if (train == null) {
      modelFile = Utils.getOption('l', options);
      if (modelFile.length() == 0)
	throw new IllegalArgumentException("Neither training set ('-t') nor serialized model file ('-l') supplied!");
      obj = SerializationHelper.readAll(modelFile);
      model = (CollectiveClassifier) obj[0];
      if (obj.length > 1)
	train = (Instances) obj[1];
      if (model.getClass() != classifier.getClass())
	throw new IllegalArgumentException("Classes differ: serialized/" + model.getClass().getName() + " != provided/" + classifier.getClass().getName());
    }
    else {
      modelFile = null;
      model     = null;
    }

    // dump file?
    tmpStr = Utils.getOption('d', options);
    if (tmpStr.length() > 0)
      dumpFile = tmpStr;
    else
      dumpFile = null;
    
    // no datasets?
    if ((train == null) && (test == null))
      throw new IllegalArgumentException("Neither train ('-t') nor test ('-T') dataset supplied!");
    
    // compatible?
    if ((train != null) && (test != null)) {
      if (!train.equalHeaders(test))
	throw new IllegalArgumentException("Train and test set not compatible:\n" + train.equalHeadersMsg(test));
    }
    
    // cross-validation?
    swap = Utils.getFlag("swap-folds", options);
    if (test == null) {
      tmpStr = Utils.getOption('x', options);
      if (tmpStr.length() == 0)
	tmpStr = "10";
      try {
	folds = Integer.parseInt(tmpStr);
      }
      catch (Exception e) {
	throw new IllegalArgumentException("Failed to parse folds ('-x') string '" + tmpStr + "'!", e);
      }
    }
    else {
      folds = 0;
    }
    
    // evaluate model
    if (folds == 0) {
      if (model == null) {
	eval = new Evaluation(train);
	classifier.setOptions(options);
	classifier.buildClassifier(train, test);
	if (dumpFile != null)
	  SerializationHelper.writeAll(dumpFile, new Object[]{classifier, new Instances(train, 0)});
	eval.evaluateModel(classifier, test);
	return eval.toSummaryString();
      }
      else {
	eval = new Evaluation(test);
	eval.evaluateModel(model, test);
	return eval.toSummaryString();
      }
    }
    else {
      if (dumpFile != null)
	throw new IllegalArgumentException("Cannot output model ('-d') when using cross-validation!");
      if (modelFile != null)
	throw new IllegalArgumentException("Cannot load model ('-l') when using cross-validation!");
      eval = new Evaluation(train);
      eval.setSwapFolds(swap);
      eval.crossValidateModel(classifier, train, folds, new Random(seed));
      return eval.toSummaryString();
    }
  }
}
