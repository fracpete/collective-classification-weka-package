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
 *    CollectiveClassifierSplitEvaluator.java
 *    Copyright (C) 1999-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.experiment;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamClass;
import java.io.Serializable;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.collective.CollectiveClassifier;
import weka.classifiers.collective.meta.YATSI;
import weka.classifiers.evaluation.AbstractEvaluationMetric;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Summarizable;
import weka.core.Utils;

/**
 * <!-- globalinfo-start --> A SplitEvaluator that produces results for a
 * collective classification scheme on a nominal class attribute.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -W &lt;class name&gt;
 *  The full class name of the classifier.
 *  eg: weka.classifiers.bayes.NaiveBayes
 * </pre>
 * 
 * <pre>
 * -C &lt;index&gt;
 *  The index of the class for which IR statistics
 *  are to be output. (default 1)
 * </pre>
 * 
 * <pre>
 * -I &lt;index&gt;
 *  The index of an attribute to output in the
 *  results. This attribute should identify an
 *  instance in order to know which instances are
 *  in the test set of a cross validation. if 0
 *  no output (default 0).
 * </pre>
 * 
 * <pre>
 * -P
 *  Add target and prediction columns to the result
 *  for each fold.
 * </pre>
 * 
 * <pre>
 * -no-size
 *  Skips the determination of sizes (train/test/classifier)
 *  (default: sizes are determined)
 * </pre>
 * 
 * <pre>
 * Options specific to classifier weka.classifiers.rules.ZeroR:
 * </pre>
 * 
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * All options after -- will be passed to the classifier.
 * 
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @version $Revision: 2027 $
 */
public class CollectiveClassifierSplitEvaluator
  extends ClassifierSplitEvaluator {

  /** for serialization */
  private static final long serialVersionUID = 892708306224595777L;

  /** The length of a key */
  public static final int KEY_SIZE = 3;

  /** The length of a result */
  public static final int RESULT_SIZE = 30;

  /** The number of IR statistics */
  public static final int NUM_IR_STATISTICS = 16;

  /** The number of averaged IR statistics */
  public static final int NUM_WEIGHTED_IR_STATISTICS = 10;

  /** The number of unweighted averaged IR statistics */
  public static final int NUM_UNWEIGHTED_IR_STATISTICS = 2;

  protected final List<AbstractEvaluationMetric> m_pluginMetrics = new ArrayList<AbstractEvaluationMetric>();
  protected int m_numPluginStatistics = 0;

  /**
   * No args constructor.
   */
  public CollectiveClassifierSplitEvaluator() {
    super();
    
    m_Template = new YATSI();
    updateOptions();
    
    List<AbstractEvaluationMetric> pluginMetrics = AbstractEvaluationMetric
        .getPluginMetrics();
    if (pluginMetrics != null) {
      for (AbstractEvaluationMetric m : pluginMetrics) {
        if (m.appliesToNominalClass()) {
          m_pluginMetrics.add(m);
          m_numPluginStatistics += m.getStatisticNames().size();
        }
      }
    }
  }

  /**
   * Returns a string describing this split evaluator
   * 
   * @return a description of the split evaluator suitable for displaying in the
   *         explorer/experimenter gui
   */
  @Override
  public String globalInfo() {
    return " A SplitEvaluator that produces results for a collective classification "
        + "scheme on a nominal class attribute.";
  }

  /**
   * Gets the data types of each of the key columns produced for a single run.
   * The number of key fields must be constant for a given SplitEvaluator.
   * 
   * @return an array containing objects of the type of each key column. The
   *         objects should be Strings, or Doubles.
   */
  @Override
  public Object[] getKeyTypes() {

    Object[] keyTypes = new Object[KEY_SIZE];
    keyTypes[0] = "";
    keyTypes[1] = "";
    keyTypes[2] = "";
    return keyTypes;
  }

  /**
   * Gets the names of each of the key columns produced for a single run. The
   * number of key fields must be constant for a given SplitEvaluator.
   * 
   * @return an array containing the name of each key column
   */
  @Override
  public String[] getKeyNames() {

    String[] keyNames = new String[KEY_SIZE];
    keyNames[0] = "Scheme";
    keyNames[1] = "Scheme_options";
    keyNames[2] = "Scheme_version_ID";
    return keyNames;
  }

  /**
   * Gets the key describing the current SplitEvaluator. For example This may
   * contain the name of the classifier used for classifier predictive
   * evaluation. The number of key fields must be constant for a given
   * SplitEvaluator.
   * 
   * @return an array of objects containing the key.
   */
  @Override
  public Object[] getKey() {

    Object[] key = new Object[KEY_SIZE];
    key[0] = m_Template.getClass().getName();
    key[1] = m_ClassifierOptions;
    key[2] = m_ClassifierVersion;
    return key;
  }

  /**
   * Gets the data types of each of the result columns produced for a single
   * run. The number of result fields must be constant for a given
   * SplitEvaluator.
   * 
   * @return an array containing objects of the type of each result column. The
   *         objects should be Strings, or Doubles.
   */
  @Override
  public Object[] getResultTypes() {
    int addm = (m_AdditionalMeasures != null) ? m_AdditionalMeasures.length : 0;
    int overall_length = RESULT_SIZE + addm;
    overall_length += NUM_IR_STATISTICS;
    overall_length += NUM_WEIGHTED_IR_STATISTICS;
    overall_length += NUM_UNWEIGHTED_IR_STATISTICS;

    if (getAttributeID() >= 0)
      overall_length += 1;
    if (getPredTargetColumn())
      overall_length += 2;

    overall_length += m_numPluginStatistics;

    Object[] resultTypes = new Object[overall_length];
    Double doub = new Double(0);
    int current = 0;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // IR stats
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // Unweighted IR stats
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // Weighted IR stats
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // Timing stats
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // sizes
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // Prediction interval statistics
    resultTypes[current++] = doub;
    resultTypes[current++] = doub;

    // ID/Targets/Predictions
    if (getAttributeID() >= 0)
      resultTypes[current++] = "";
    if (getPredTargetColumn()) {
      resultTypes[current++] = "";
      resultTypes[current++] = "";
    }

    // Classifier defined extras
    resultTypes[current++] = "";

    // add any additional measures
    for (int i = 0; i < addm; i++) {
      resultTypes[current++] = doub;
    }

    // plugin metrics
    for (int i = 0; i < m_numPluginStatistics; i++) {
      resultTypes[current++] = doub;
    }

    if (current != overall_length) {
      throw new Error("ResultTypes didn't fit RESULT_SIZE");
    }
    return resultTypes;
  }

  /**
   * Gets the names of each of the result columns produced for a single run. The
   * number of result fields must be constant for a given SplitEvaluator.
   * 
   * @return an array containing the name of each result column
   */
  @Override
  public String[] getResultNames() {
    int addm = (m_AdditionalMeasures != null) ? m_AdditionalMeasures.length : 0;
    int overall_length = RESULT_SIZE + addm;
    overall_length += NUM_IR_STATISTICS;
    overall_length += NUM_WEIGHTED_IR_STATISTICS;
    overall_length += NUM_UNWEIGHTED_IR_STATISTICS;
    if (getAttributeID() >= 0)
      overall_length += 1;
    if (getPredTargetColumn())
      overall_length += 2;

    overall_length += m_pluginMetrics.size();

    String[] resultNames = new String[overall_length];
    int current = 0;
    resultNames[current++] = "Number_of_training_instances";
    resultNames[current++] = "Number_of_testing_instances";

    // Basic performance stats - right vs wrong
    resultNames[current++] = "Number_correct";
    resultNames[current++] = "Number_incorrect";
    resultNames[current++] = "Number_unclassified";
    resultNames[current++] = "Percent_correct";
    resultNames[current++] = "Percent_incorrect";
    resultNames[current++] = "Percent_unclassified";
    resultNames[current++] = "Kappa_statistic";

    // Sensitive stats - certainty of predictions
    resultNames[current++] = "Mean_absolute_error";
    resultNames[current++] = "Root_mean_squared_error";
    resultNames[current++] = "Relative_absolute_error";
    resultNames[current++] = "Root_relative_squared_error";

    // SF stats
    resultNames[current++] = "SF_prior_entropy";
    resultNames[current++] = "SF_scheme_entropy";
    resultNames[current++] = "SF_entropy_gain";
    resultNames[current++] = "SF_mean_prior_entropy";
    resultNames[current++] = "SF_mean_scheme_entropy";
    resultNames[current++] = "SF_mean_entropy_gain";

    // K&B stats
    resultNames[current++] = "KB_information";
    resultNames[current++] = "KB_mean_information";
    resultNames[current++] = "KB_relative_information";

    // IR stats
    resultNames[current++] = "True_positive_rate";
    resultNames[current++] = "Num_true_positives";
    resultNames[current++] = "False_positive_rate";
    resultNames[current++] = "Num_false_positives";
    resultNames[current++] = "True_negative_rate";
    resultNames[current++] = "Num_true_negatives";
    resultNames[current++] = "False_negative_rate";
    resultNames[current++] = "Num_false_negatives";
    resultNames[current++] = "IR_precision";
    resultNames[current++] = "IR_recall";
    resultNames[current++] = "F_measure";
    resultNames[current++] = "Matthews_correlation";
    resultNames[current++] = "Area_under_ROC";
    resultNames[current++] = "Area_under_PRC";

    // Weighted IR stats
    resultNames[current++] = "Weighted_avg_true_positive_rate";
    resultNames[current++] = "Weighted_avg_false_positive_rate";
    resultNames[current++] = "Weighted_avg_true_negative_rate";
    resultNames[current++] = "Weighted_avg_false_negative_rate";
    resultNames[current++] = "Weighted_avg_IR_precision";
    resultNames[current++] = "Weighted_avg_IR_recall";
    resultNames[current++] = "Weighted_avg_F_measure";
    resultNames[current++] = "Weighted_avg_matthews_correlation";
    resultNames[current++] = "Weighted_avg_area_under_ROC";
    resultNames[current++] = "Weighted_avg_area_under_PRC";

    // Unweighted IR stats
    resultNames[current++] = "Unweighted_macro_avg_F_measure";
    resultNames[current++] = "Unweighted_micro_avg_F_measure";

    // Timing stats
    resultNames[current++] = "Elapsed_Time_training";
    resultNames[current++] = "Elapsed_Time_testing";
    resultNames[current++] = "UserCPU_Time_training";
    resultNames[current++] = "UserCPU_Time_testing";

    // sizes
    resultNames[current++] = "Serialized_Model_Size";
    resultNames[current++] = "Serialized_Train_Set_Size";
    resultNames[current++] = "Serialized_Test_Set_Size";

    // Prediction interval statistics
    resultNames[current++] = "Coverage_of_Test_Cases_By_Regions";
    resultNames[current++] = "Size_of_Predicted_Regions";

    // ID/Targets/Predictions
    if (getAttributeID() >= 0)
      resultNames[current++] = "Instance_ID";
    if (getPredTargetColumn()) {
      resultNames[current++] = "Targets";
      resultNames[current++] = "Predictions";
    }

    // Classifier defined extras
    resultNames[current++] = "Summary";
    // add any additional measures
    for (int i = 0; i < addm; i++) {
      resultNames[current++] = m_AdditionalMeasures[i];
    }

    for (AbstractEvaluationMetric m : m_pluginMetrics) {
      List<String> statNames = m.getStatisticNames();
      for (String s : statNames) {
        resultNames[current++] = s;
      }
    }

    if (current != overall_length) {
      throw new Error("ResultNames didn't fit RESULT_SIZE");
    }
    return resultNames;
  }

  /**
   * Gets the results for the supplied train and test datasets. Now performs a
   * deep copy of the classifier before it is built and evaluated (just in case
   * the classifier is not initialized properly in buildClassifier()).
   * 
   * @param train the training Instances.
   * @param test the testing Instances.
   * @return the results stored in an array. The objects stored in the array may
   *         be Strings, Doubles, or null (for the missing value).
   * @throws Exception if a problem occurs while getting the results
   */
  @Override
  public Object[] getResult(Instances train, Instances test) throws Exception {

    if (train.classAttribute().type() != Attribute.NOMINAL) {
      throw new Exception("Class attribute is not nominal!");
    }
    if (m_Template == null) {
      throw new Exception("No classifier has been specified");
    }
    int addm = (m_AdditionalMeasures != null) ? m_AdditionalMeasures.length : 0;
    int overall_length = RESULT_SIZE + addm;
    overall_length += NUM_IR_STATISTICS;
    overall_length += NUM_WEIGHTED_IR_STATISTICS;
    overall_length += NUM_UNWEIGHTED_IR_STATISTICS;
    if (getAttributeID() >= 0)
      overall_length += 1;
    if (getPredTargetColumn())
      overall_length += 2;

    overall_length += m_pluginMetrics.size();

    ThreadMXBean thMonitor = ManagementFactory.getThreadMXBean();
    boolean canMeasureCPUTime = thMonitor.isThreadCpuTimeSupported();
    if (canMeasureCPUTime && !thMonitor.isThreadCpuTimeEnabled())
      thMonitor.setThreadCpuTimeEnabled(true);

    Object[] result = new Object[overall_length];
    Evaluation eval = new Evaluation(train);
    m_Classifier = AbstractClassifier.makeCopy(m_Template);
    double[] predictions;
    long thID = Thread.currentThread().getId();
    long CPUStartTime = -1, trainCPUTimeElapsed = -1, testCPUTimeElapsed = -1, trainTimeStart, trainTimeElapsed, testTimeStart, testTimeElapsed;

    // training classifier
    trainTimeStart = System.currentTimeMillis();
    if (canMeasureCPUTime)
      CPUStartTime = thMonitor.getThreadUserTime(thID);
    if (m_Classifier instanceof CollectiveClassifier)
      ((CollectiveClassifier) m_Classifier).buildClassifier(train, test);
    else
      m_Classifier.buildClassifier(train);
    if (canMeasureCPUTime)
      trainCPUTimeElapsed = thMonitor.getThreadUserTime(thID) - CPUStartTime;
    trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;

    // testing classifier
    testTimeStart = System.currentTimeMillis();
    if (canMeasureCPUTime)
      CPUStartTime = thMonitor.getThreadUserTime(thID);
    predictions = eval.evaluateModel(m_Classifier, test);
    if (canMeasureCPUTime)
      testCPUTimeElapsed = thMonitor.getThreadUserTime(thID) - CPUStartTime;
    testTimeElapsed = System.currentTimeMillis() - testTimeStart;
    thMonitor = null;

    m_result = eval.toSummaryString();
    // The results stored are all per instance -- can be multiplied by the
    // number of instances to get absolute numbers
    int current = 0;
    result[current++] = new Double(train.numInstances());
    result[current++] = new Double(eval.numInstances());
    result[current++] = new Double(eval.correct());
    result[current++] = new Double(eval.incorrect());
    result[current++] = new Double(eval.unclassified());
    result[current++] = new Double(eval.pctCorrect());
    result[current++] = new Double(eval.pctIncorrect());
    result[current++] = new Double(eval.pctUnclassified());
    result[current++] = new Double(eval.kappa());

    result[current++] = new Double(eval.meanAbsoluteError());
    result[current++] = new Double(eval.rootMeanSquaredError());
    result[current++] = new Double(eval.relativeAbsoluteError());
    result[current++] = new Double(eval.rootRelativeSquaredError());

    result[current++] = new Double(eval.SFPriorEntropy());
    result[current++] = new Double(eval.SFSchemeEntropy());
    result[current++] = new Double(eval.SFEntropyGain());
    result[current++] = new Double(eval.SFMeanPriorEntropy());
    result[current++] = new Double(eval.SFMeanSchemeEntropy());
    result[current++] = new Double(eval.SFMeanEntropyGain());

    // K&B stats
    result[current++] = new Double(eval.KBInformation());
    result[current++] = new Double(eval.KBMeanInformation());
    result[current++] = new Double(eval.KBRelativeInformation());

    // IR stats
    result[current++] = new Double(eval.truePositiveRate(getClassForIRStatistics()));
    result[current++] = new Double(eval.numTruePositives(getClassForIRStatistics()));
    result[current++] = new Double(eval.falsePositiveRate(getClassForIRStatistics()));
    result[current++] = new Double(eval.numFalsePositives(getClassForIRStatistics()));
    result[current++] = new Double(eval.trueNegativeRate(getClassForIRStatistics()));
    result[current++] = new Double(eval.numTrueNegatives(getClassForIRStatistics()));
    result[current++] = new Double(eval.falseNegativeRate(getClassForIRStatistics()));
    result[current++] = new Double(eval.numFalseNegatives(getClassForIRStatistics()));
    result[current++] = new Double(eval.precision(getClassForIRStatistics()));
    result[current++] = new Double(eval.recall(getClassForIRStatistics()));
    result[current++] = new Double(eval.fMeasure(getClassForIRStatistics()));
    result[current++] = new Double(
        eval.matthewsCorrelationCoefficient(getClassForIRStatistics()));
    result[current++] = new Double(eval.areaUnderROC(getClassForIRStatistics()));
    result[current++] = new Double(eval.areaUnderPRC(getClassForIRStatistics()));

    // Weighted IR stats
    result[current++] = new Double(eval.weightedTruePositiveRate());
    result[current++] = new Double(eval.weightedFalsePositiveRate());
    result[current++] = new Double(eval.weightedTrueNegativeRate());
    result[current++] = new Double(eval.weightedFalseNegativeRate());
    result[current++] = new Double(eval.weightedPrecision());
    result[current++] = new Double(eval.weightedRecall());
    result[current++] = new Double(eval.weightedFMeasure());
    result[current++] = new Double(eval.weightedMatthewsCorrelation());
    result[current++] = new Double(eval.weightedAreaUnderROC());
    result[current++] = new Double(eval.weightedAreaUnderPRC());

    // Unweighted IR stats
    result[current++] = new Double(eval.unweightedMacroFmeasure());
    result[current++] = new Double(eval.unweightedMicroFmeasure());

    // Timing stats
    result[current++] = new Double(trainTimeElapsed / 1000.0);
    result[current++] = new Double(testTimeElapsed / 1000.0);
    if (canMeasureCPUTime) {
      result[current++] = new Double((trainCPUTimeElapsed / 1000000.0) / 1000.0);
      result[current++] = new Double((testCPUTimeElapsed / 1000000.0) / 1000.0);
    } else {
      result[current++] = new Double(Utils.missingValue());
      result[current++] = new Double(Utils.missingValue());
    }

    // sizes
    if (getNoSizeDetermination()) {
      result[current++] = -1.0;
      result[current++] = -1.0;
      result[current++] = -1.0;
    } else {
      ByteArrayOutputStream bastream = new ByteArrayOutputStream();
      ObjectOutputStream oostream = new ObjectOutputStream(bastream);
      oostream.writeObject(m_Classifier);
      result[current++] = new Double(bastream.size());
      bastream = new ByteArrayOutputStream();
      oostream = new ObjectOutputStream(bastream);
      oostream.writeObject(train);
      result[current++] = new Double(bastream.size());
      bastream = new ByteArrayOutputStream();
      oostream = new ObjectOutputStream(bastream);
      oostream.writeObject(test);
      result[current++] = new Double(bastream.size());
    }

    // Prediction interval statistics
    result[current++] = new Double(eval.coverageOfTestCasesByPredictedRegions());
    result[current++] = new Double(eval.sizeOfPredictedRegions());

    // IDs
    if (getAttributeID() >= 0) {
      String idsString = "";
      if (test.attribute(getAttributeID()).isNumeric()) {
        if (test.numInstances() > 0)
          idsString += test.instance(0).value(getAttributeID());
        for (int i = 1; i < test.numInstances(); i++) {
          idsString += "|" + test.instance(i).value(getAttributeID());
        }
      } else {
        if (test.numInstances() > 0)
          idsString += test.instance(0).stringValue(getAttributeID());
        for (int i = 1; i < test.numInstances(); i++) {
          idsString += "|" + test.instance(i).stringValue(getAttributeID());
        }
      }
      result[current++] = idsString;
    }

    if (getPredTargetColumn()) {
      if (test.classAttribute().isNumeric()) {
        // Targets
        if (test.numInstances() > 0) {
          String targetsString = "";
          targetsString += test.instance(0).value(test.classIndex());
          for (int i = 1; i < test.numInstances(); i++) {
            targetsString += "|" + test.instance(i).value(test.classIndex());
          }
          result[current++] = targetsString;
        }

        // Predictions
        if (predictions.length > 0) {
          String predictionsString = "";
          predictionsString += predictions[0];
          for (int i = 1; i < predictions.length; i++) {
            predictionsString += "|" + predictions[i];
          }
          result[current++] = predictionsString;
        }
      } else {
        // Targets
        if (test.numInstances() > 0) {
          String targetsString = "";
          targetsString += test.instance(0).stringValue(test.classIndex());
          for (int i = 1; i < test.numInstances(); i++) {
            targetsString += "|"
                + test.instance(i).stringValue(test.classIndex());
          }
          result[current++] = targetsString;
        }

        // Predictions
        if (predictions.length > 0) {
          String predictionsString = "";
          predictionsString += test.classAttribute()
              .value((int) predictions[0]);
          for (int i = 1; i < predictions.length; i++) {
            predictionsString += "|"
                + test.classAttribute().value((int) predictions[i]);
          }
          result[current++] = predictionsString;
        }
      }
    }

    if (m_Classifier instanceof Summarizable) {
      result[current++] = ((Summarizable) m_Classifier).toSummaryString();
    } else {
      result[current++] = null;
    }

    for (int i = 0; i < addm; i++) {
      if (m_doesProduce[i]) {
        try {
          double dv = ((AdditionalMeasureProducer) m_Classifier)
              .getMeasure(m_AdditionalMeasures[i]);
          if (!Utils.isMissingValue(dv)) {
            Double value = new Double(dv);
            result[current++] = value;
          } else {
            result[current++] = null;
          }
        } catch (Exception ex) {
          System.err.println(ex);
        }
      } else {
        result[current++] = null;
      }
    }

    // get the actual metrics from the evaluation object
    List<AbstractEvaluationMetric> metrics = eval.getPluginMetrics();
    if (metrics != null) {
      for (AbstractEvaluationMetric m : metrics) {
        if (m.appliesToNominalClass()) {
          List<String> statNames = m.getStatisticNames();
          for (String s : statNames) {
            result[current++] = new Double(m.getStatistic(s));
          }
        }
      }
    }

    if (current != overall_length) {
      throw new Error("Results didn't fit RESULT_SIZE");
    }
    return result;
  }

  /**
   * Updates the options that the current classifier is using.
   */
  @Override
  protected void updateOptions() {

    if (m_Template instanceof OptionHandler) {
      m_ClassifierOptions = Utils.joinOptions(((OptionHandler) m_Template)
          .getOptions());
    } else {
      m_ClassifierOptions = "";
    }
    if (m_Template instanceof Serializable) {
      ObjectStreamClass obs = ObjectStreamClass.lookup(m_Template.getClass());
      m_ClassifierVersion = "" + obs.getSerialVersionUID();
    } else {
      m_ClassifierVersion = "";
    }
  }

  /**
   * Set the Classifier to use, given it's class name. A new classifier will be
   * instantiated.
   * 
   * @param newClassifierName the Classifier class name.
   * @throws Exception if the class name is invalid.
   */
  @Override
  public void setClassifierName(String newClassifierName) throws Exception {

    try {
      setClassifier((Classifier) Class.forName(newClassifierName).newInstance());
    } catch (Exception ex) {
      throw new Exception("Can't find Classifier with class name: "
          + newClassifierName);
    }
  }

  /**
   * Returns a text description of the split evaluator.
   * 
   * @return a text description of the split evaluator.
   */
  @Override
  public String toString() {

    String result = "CollectiveClassifierSplitEvaluator: ";
    if (m_Template == null) {
      return result + "<null> classifier";
    }
    return result + m_Template.getClass().getName() + " " + m_ClassifierOptions
        + "(version " + m_ClassifierVersion + ")";
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 2027 $");
  }
}
