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
 * AdvancedCollective.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.meta;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.collective.util.CollectiveInstances;
import weka.classifiers.collective.util.FlipHistory;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;

/**
 <!-- globalinfo-start -->
 * Uses collective classification (cf. transduction, semi-supersived learning), i.e., it uses additionally the data given in the test file (with random labels at first) to get the most out of a small training set. It is attempted to improve the initial random labels through several iterations, whereas a restart reinitializes all labels again randomly.<br/>
 * To avoid overfitting it splits the test set into two parts rather than using it completely for training and uses the other half for testing (and vice versa).
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -I &lt;num&gt;
 *  Number of iterations.
 *  (default 10)</pre>
 *
 * <pre> -R &lt;num&gt;
 *  Number of restarts.
 *  (default 10)</pre>
 *
 * <pre> -log
 *  Creates logs in the tmp directory for all kinds of internal data.
 *  Use only for debugging purposes!
 * </pre>
 *
 * <pre> -U
 *  Updates also the labels of the training set.
 * </pre>
 *
 * <pre> -eval &lt;num&gt;
 *  The type of evaluation to use (0 = Randomwalk/Last model used for
 *  prediction, 1=Randomwalk/Best model used for prediction,
 *  2=Hillclimbing).
 * </pre>
 *
 * <pre> -compare &lt;num&gt;
 *  The type of comparisong used for comparing models.
 *  (0=overall RMS, 1=RMS on train set, 2=RMS on test set,
 *  3=Accuracy on train set)
 * </pre>
 *
 * <pre> -flipper "&lt;classname [parameters]&gt;"
 *  The flipping algorithm (and optional parameters) to use for
 *  flipping labels.
 * </pre>
 *
 * <pre> -folds &lt;folds&gt;
 *  The number of folds for splitting the training set into
 *  train and test set. The first fold is always the training
 *  set. With '-V' you can invert this, i.e., instead of 20/80
 *  for 5 folds you'll get 80/20.
 *  (default 5)</pre>
 *
 * <pre> -V
 *  Inverts the fold selection, i.e., instead of using the first
 *  fold for the training set it is used for test set and the
 *  remaining folds for training.</pre>
 *
 * <pre> -verbose
 *  Whether to print some more information during building the
 *  classifier.
 *  (default is off)</pre>
 *
 * <pre> -verbose
 *  Whether to print some more information during building the
 *  classifier.
 *  (default is off)</pre>
 *
 * <pre> -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)</pre>
 *
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 *
 * <pre> -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.J48)</pre>
 *
 * <pre>
 * Options specific to classifier weka.classifiers.trees.J48:
 * </pre>
 *
 * <pre> -U
 *  Use unpruned tree.</pre>
 *
 * <pre> -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)</pre>
 *
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)</pre>
 *
 * <pre> -R
 *  Use reduced error pruning.</pre>
 *
 * <pre> -N &lt;number of folds&gt;
 *  Set number of folds for reduced error
 *  pruning. One fold is used as pruning set.
 *  (default 3)</pre>
 *
 * <pre> -B
 *  Use binary splits only.</pre>
 *
 * <pre> -S
 *  Don't perform subtree raising.</pre>
 *
 * <pre> -L
 *  Do not clean up after the tree has been built.</pre>
 *
 * <pre> -A
 *  Laplace smoothing for predicted probabilities.</pre>
 *
 * <pre> -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).</pre>
 *
 <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.<p/>
 *
 * @author Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class AdvancedCollective
  extends SimpleCollective {

  /** for serialization */
  private static final long serialVersionUID = -6257120935331668113L;

  /** The two different classifiers, working on the two different datasets */
  protected Classifier m_Classifier2;

  /** the new training set */
  protected Instances m_TrainsetNew2;

  /** the number of flipped labels in the first training set */
  protected double m_FlippedLabels1;

  /** the number of flipped labels in the second training set */
  protected double m_FlippedLabels2;

  /** the flipping history */
  protected FlipHistory m_FlipHistory2;

  /**
   * performs initialization of members
   */
  @Override
  protected void initializeMembers() {
    super.initializeMembers();

    m_Classifier     = new weka.classifiers.trees.J48();
    m_Classifier2    = new weka.classifiers.trees.J48();
    m_TrainsetNew2   = null;
    m_FlippedLabels1 = 0;
    m_FlippedLabels2 = 0;
    m_FlipHistory2   = null;
  }

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  @Override
  public String globalInfo() {
    return    "Uses collective classification (cf. transduction, semi-supersived "
    + "learning), i.e., it uses additionally the data given in the test file "
    + "(with random labels at first) to get the most out of a small training "
    + "set. It is attempted to improve the initial random labels through "
    + "several iterations, whereas a restart reinitializes all labels again "
    + "randomly.\n"
    + "To avoid overfitting it splits the test set into two parts "
    + "rather than using it completely for training and uses the other half "
    + "for testing (and vice versa).";
  }

  /**
   * Set the base learner. Creates copies of the given classifier for the two
   * classifiers used internally. <br/>
   * Note: also unsets the flag whether the classifier has been built so far.
   *
   * @param newClassifier the classifier to use.
   * @see #m_ClassifierBuilt
   */
  public void setClassifier(Classifier newClassifier) {
    super.setClassifier(newClassifier);

    try {
      m_Classifier2 = AbstractClassifier.makeCopies(newClassifier, 1)[0];
    }
    catch (Exception e) {
      e.printStackTrace();
      m_Classifier  = new weka.classifiers.trees.J48();
      m_Classifier2 = new weka.classifiers.trees.J48();
    }
  }

  /**
   * internal function for determining the class distribution for an instance,
   * will be overridden by derived classes. For more details about the returned
   * array, see <code>Classifier.distributionForInstance(Instance)</code>.
   *
   * @param instance	the instance to get the distribution for
   * @return		the distribution for the given instance
   * @see 		weka.classifiers.Classifier#distributionForInstance(Instance)
   * @throws Exception	if something goes wrong
   */
  @Override
  protected double[] getDistribution(Instance instance) throws Exception {
    double[]     dist1;
    double[]     dist2;
    double[]     result;
    int          i;

    dist1  = m_Classifier.distributionForInstance(instance);
    dist2  = m_Classifier2.distributionForInstance(instance);
    result = new double[dist1.length];

    for (i = 0; i < dist1.length; i++)
      result[i] = (dist1[i] + dist2[i]) / 2;

    return result;
  }

  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   *
   * <pre> -I &lt;num&gt;
   *  Number of iterations.
   *  (default 10)</pre>
   *
   * <pre> -R &lt;num&gt;
   *  Number of restarts.
   *  (default 10)</pre>
   *
   * <pre> -log
   *  Creates logs in the tmp directory for all kinds of internal data.
   *  Use only for debugging purposes!
   * </pre>
   *
   * <pre> -U
   *  Updates also the labels of the training set.
   * </pre>
   *
   * <pre> -eval &lt;num&gt;
   *  The type of evaluation to use (0 = Randomwalk/Last model used for
   *  prediction, 1=Randomwalk/Best model used for prediction,
   *  2=Hillclimbing).
   * </pre>
   *
   * <pre> -compare &lt;num&gt;
   *  The type of comparisong used for comparing models.
   *  (0=overall RMS, 1=RMS on train set, 2=RMS on test set,
   *  3=Accuracy on train set)
   * </pre>
   *
   * <pre> -flipper "&lt;classname [parameters]&gt;"
   *  The flipping algorithm (and optional parameters) to use for
   *  flipping labels.
   * </pre>
   *
   * <pre> -folds &lt;folds&gt;
   *  The number of folds for splitting the training set into
   *  train and test set. The first fold is always the training
   *  set. With '-V' you can invert this, i.e., instead of 20/80
   *  for 5 folds you'll get 80/20.
   *  (default 5)</pre>
   *
   * <pre> -V
   *  Inverts the fold selection, i.e., instead of using the first
   *  fold for the training set it is used for test set and the
   *  remaining folds for training.</pre>
   *
   * <pre> -verbose
   *  Whether to print some more information during building the
   *  classifier.
   *  (default is off)</pre>
   *
   * <pre> -verbose
   *  Whether to print some more information during building the
   *  classifier.
   *  (default is off)</pre>
   *
   * <pre> -S &lt;num&gt;
   *  Random number seed.
   *  (default 1)</pre>
   *
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   *
   * <pre> -W
   *  Full name of base classifier.
   *  (default: weka.classifiers.trees.J48)</pre>
   *
   * <pre>
   * Options specific to classifier weka.classifiers.trees.J48:
   * </pre>
   *
   * <pre> -U
   *  Use unpruned tree.</pre>
   *
   * <pre> -C &lt;pruning confidence&gt;
   *  Set confidence threshold for pruning.
   *  (default 0.25)</pre>
   *
   * <pre> -M &lt;minimum number of instances&gt;
   *  Set minimum number of instances per leaf.
   *  (default 2)</pre>
   *
   * <pre> -R
   *  Use reduced error pruning.</pre>
   *
   * <pre> -N &lt;number of folds&gt;
   *  Set number of folds for reduced error
   *  pruning. One fold is used as pruning set.
   *  (default 3)</pre>
   *
   * <pre> -B
   *  Use binary splits only.</pre>
   *
   * <pre> -S
   *  Don't perform subtree raising.</pre>
   *
   * <pre> -L
   *  Do not clean up after the tree has been built.</pre>
   *
   * <pre> -A
   *  Laplace smoothing for predicted probabilities.</pre>
   *
   * <pre> -Q &lt;seed&gt;
   *  Seed for random data shuffling (default 1).</pre>
   *
   <!-- options-end -->
   *
   * Options after -- are passed to the designated classifier.<p/>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    super.setOptions(options);

    ((OptionHandler) m_Classifier2).setOptions(((OptionHandler) m_Classifier).getOptions());
  }

  /**
   * builds the necessary CollectiveInstances from the given Instances
   * @throws Exception if anything goes wrong
   */
  @Override
  protected void generateSets() throws Exception {
    int       i;
    int       count;

    super.generateSets();

    // determine the middle of the test instances
    count = (int) m_Testset.numInstances() / 2;

    m_TrainsetNew = new Instances(m_Trainset);
    for (i = 0; i < count; i++)
      m_TrainsetNew.add(m_Testset.instance(i));

    m_TrainsetNew2 = new Instances(m_Trainset);
    for (i = count; i < m_Testset.numInstances(); i++)
      m_TrainsetNew2.add(m_Testset.instance(i));

    m_FlipHistory2 = new FlipHistory(m_TrainsetNew2);
  }

  /**
   * initializes the labels of the CollectiveInstances
   * @throws Exception if initialization fails
   */
  @Override
  public void initializeLabels() throws Exception {
    m_CollectiveInstances.initializeLabels(
        m_Trainset, m_TrainsetNew,
        m_Trainset.numInstances(),
        m_TrainsetNew.numInstances() - m_Trainset.numInstances());
    m_FlippedLabels1 = m_CollectiveInstances.getFlippedLabels();

    m_CollectiveInstances.initializeLabels(
        m_Trainset, m_TrainsetNew2,
        m_Trainset.numInstances(),
        m_TrainsetNew2.numInstances() - m_Trainset.numInstances());
    m_FlippedLabels2 = m_CollectiveInstances.getFlippedLabels();
  }

  /**
   * flips the labels of the CollectiveInstances
   * @throws Exception if flipping fails
   */
  @Override
  public void flipLabels() throws Exception {
    int       from;
    int       count;

    // first
    if (getUpdateTraining()) {
      from  = 0;
      count = m_TrainsetNew.numInstances();
    }
    else {
      from  = m_Trainset.numInstances();
      count = m_TrainsetNew.numInstances() - m_Trainset.numInstances();
    }
    if (m_EvaluationType == CollectiveInstances.EVAL_HILLCLIMBING)
      m_CollectiveInstances.flipLabels(
          getBestModel(), m_TrainsetNew, from, count, m_FlipHistory);
    else
      m_CollectiveInstances.flipLabels(
          this, m_TrainsetNew, from, count, m_FlipHistory);
    m_FlippedLabels1 = m_CollectiveInstances.getFlippedLabels();

    // second
    if (getUpdateTraining()) {
      from  = 0;
      count = m_TrainsetNew2.numInstances();
    }
    else {
      from  = m_Trainset.numInstances();
      count = m_TrainsetNew2.numInstances() - m_Trainset.numInstances();
    }
    if (m_EvaluationType == CollectiveInstances.EVAL_HILLCLIMBING)
      m_CollectiveInstances.flipLabels(
          getBestModel(), m_TrainsetNew2, from, count, m_FlipHistory);
    else
      m_CollectiveInstances.flipLabels(
          this, m_TrainsetNew2, from, count, m_FlipHistory2);
    m_FlippedLabels2 = m_CollectiveInstances.getFlippedLabels();
  }

  /**
   * performs the actual building of the classifier
   * @throws Exception if building fails
   */
  @Override
  protected void buildClassifier() throws Exception {
    m_Classifier.buildClassifier(m_TrainsetNew);
    m_Classifier2.buildClassifier(m_TrainsetNew2);
  }

  /**
   * calculates the RMS for test and train set
   *
   * @throws Exception	if something goes wrong
   */
  @Override
  protected void calculateRMS() throws Exception {
    double[]      rms1;
    double[]      rms2;

    rms1 = CollectiveInstances.calculateRMS(
               m_Classifier, m_Trainset,
               new Instances(
                 m_TrainsetNew,
                 m_Trainset.numInstances(),
                 m_TrainsetNew.numInstances()
                   - m_Trainset.numInstances()),
               m_TestsetOriginal);
    rms2 = CollectiveInstances.calculateRMS(
                m_Classifier2, m_Trainset,
                new Instances(
                  m_TrainsetNew2,
                  m_Trainset.numInstances(),
                  m_TrainsetNew2.numInstances()
                    - m_Trainset.numInstances()),
                m_TestsetOriginal);

    m_RMS             = (rms1[0] + rms2[0]) / 2.0;
    m_RMSTrain        = (rms1[1] + rms2[1]) / 2.0;
    m_RMSTest         = (rms1[2] + rms2[2]) / 2.0;
    m_RMSTestOriginal = (rms1[3] + rms2[3]) / 2.0;

    if (getVerbose())
      System.out.println(   "\nRMSTest: "   + m_RMSTest + ", "
                          + "RMSTestOrig: " + m_RMSTestOriginal + ", "
                          + "RMSTrain: "    + m_RMSTrain + ", "
                          + "RMS: "         + m_RMS);
  }

  /**
   * calculates the accuracy for original test and train set
   *
   * @throws Exception	if something goes wrong
   */
  @Override
  protected void calculateAccuracy() throws Exception {
    double[]      acc1;
    double[]      acc2;

    acc1 = CollectiveInstances.calculateAccuracy(
              m_Classifier,  m_Trainset, m_TestsetOriginal);
    acc2 = CollectiveInstances.calculateAccuracy(
              m_Classifier2, m_Trainset, m_TestsetOriginal);

    m_AccTrain        = (acc1[0] + acc2[0]) / 2.0;
    m_AccTestOriginal = (acc1[1] + acc2[1]) / 2.0;

    if (getVerbose())
      System.out.println(   "\nAccTrain: "   + m_AccTrain + ", "
                          + "AccTestOrig: " + m_AccTestOriginal);
  }

  /**
   * returns the percentage of flipped labels
   *
   * @return		the percentage
   * @see 		CollectiveInstances#getFlippedLabels()
   */
  @Override
  protected double getFlippedLabels() {
    return (m_FlippedLabels1 + m_FlippedLabels2) / 2.0;
  }

  /**
   * returns the best model as string representation. derived classes have to
   * add additional information here, like printing the classifier etc.
   *
   * @return		the string representation of the best model
   */
  @Override
  protected String toStringModel() {
    StringBuffer    text;

    text = new StringBuffer();
    text.append(super.toStringModel());
    text.append("\n");
    text.append("2. Classifier\n");
    text.append("-------------\n");
    text.append("\n");
    text.append(m_Classifier2.toString());
    text.append("\n\n");

    return text.toString();
  }

  /**
   * Main method for testing this class.
   *
   * @param args the options
   */
  public static void main(String[] args) {
    runClassifier(new AdvancedCollective(), args);
  }
}
