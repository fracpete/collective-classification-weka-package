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
 * CollectiveIBk.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.lazy;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.collective.CollectiveRandomizableClassifier;
import weka.classifiers.collective.lazy.ibk.Neighbors;
import weka.classifiers.collective.util.CollectiveHelper;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.classifiers.lazy.IBk;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 <!-- globalinfo-start -->
 * Uses IBk to determine the optimal K for the neighborhood on the training set. This K is used to build for each instance of the test set a neighborhood, consisting of instances from test AND training set.<br/>
 * Majority vote is used to determine the class label for an instance, based on its neighborhood (in case of ties the first encountered class is taken). Sooner or later all labels of the test set are determined.<br/>
 * An instance that is presented for classification is then only looked up and the determined label returned.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
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
 * <pre> -insight
 *  Whether to use the labels of the original test set for more
 *  statistics (not used for learning!).
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
 * <pre> -naive
 *  Uses a sorted list (ordered according to distance) instead of the
 *  KDTree for finding the neighbors.
 *  (default is KDTree)</pre>
 * 
 * <pre> -I
 *  Weight neighbours by the inverse of their distance
 *  (use when k &gt; 1)</pre>
 * 
 * <pre> -F
 *  Weight neighbours by 1 - their distance
 *  (use when k &gt; 1)</pre>
 * 
 * <pre> -K &lt;number of neighbors&gt;
 *  Number of nearest neighbours (k) used in classification.
 *  (Default = 1)</pre>
 * 
 * <pre> -A
 *  The nearest neighbour search algorithm to use (default: LinearNN).
 * </pre>
 * 
 <!-- options-end -->
 *
 * @author Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $ 
 */
public class CollectiveIBk
  extends CollectiveRandomizableClassifier {

  /** for serialization */
  private static final long serialVersionUID = 2958216762402427687L;

  /**
   * the classifier internally used for classification
   * @see IBk
   */
  protected IBk m_Classifier;

  /** the original KNN */
  protected int m_KNN;
  
  /** the determined KNN */
  protected int m_KNNdetermined;
  
  /** the neighbors (from train + test) for all the instances in the test set */
  protected Neighbors[] m_NeighborsTestset;
  
  /** copy of the original training dataset */
  protected Instances m_TrainsetNew;
  
  /** copy of the original test dataset */
  protected Instances m_TestsetNew;

  /** for processing the missing values */
  protected ReplaceMissingValues m_Missing;

  /** whether to use the naive search, searching a list, instead of KDTree */
  public boolean m_UseNaiveSearch;

  /** 
   * stores the sorted array of labeled test-instance's - used for determining
   * the class label for a test instance
   * @see #getDistribution(Instance)
   */
  protected Instance[] m_LabeledTestset;

  /**
   * performs initialization of members
   */
  @Override
  protected void initializeMembers() {
    super.initializeMembers();

    m_KNNdetermined    = -1;
    m_NeighborsTestset = null;
    m_TrainsetNew      = null;
    m_TestsetNew       = null;
    m_UseNaiveSearch   = false;
    m_LabeledTestset   = null;
    m_Missing          = new ReplaceMissingValues();
    
    m_Classifier = new IBk();
    m_Classifier.setKNN(10);
    m_Classifier.setCrossValidate(true);
    m_Classifier.setWindowSize(0);
    m_Classifier.setMeanSquared(false);
    
    m_KNN = m_Classifier.getKNN();
    
    m_AdditionalMeasures.add("measureDeterminedKNN");
  }

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return    "Uses IBk to determine the optimal K for the neighborhood on the "
            + "training set. This K is used to build for each instance of the "
            + "test set a neighborhood, consisting of instances from test AND "
            + "training set.\n"
            + "Majority vote is used to determine the class label for an "
            + "instance, based on its neighborhood (in case of ties the first "
            + "encountered class is taken). Sooner or later all labels of the "
            + "test set are determined.\n"
            + "An instance that is presented for classification is then only "
            + "looked up and the determined label returned.";
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String KNNTipText() {
    return m_Classifier.KNNTipText();
  }
  
  /**
   * Set the number of neighbours the learner is to use.
   *
   * @param k the number of neighbours.
   */
  public void setKNN(int k) {
    m_Classifier.setKNN(k);
  }

  /**
   * Gets the number of neighbours the learner will use.
   *
   * @return the number of neighbours.
   */
  public int getKNN() {
    return m_Classifier.getKNN();
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String distanceWeightingTipText() {
    return m_Classifier.distanceWeightingTipText();
  }
  
  /**
   * Gets the distance weighting method used. Will be one of
   * WEIGHT_NONE, WEIGHT_INVERSE, or WEIGHT_SIMILARITY
   *
   * @return the distance weighting method used.
   * @see IBk#WEIGHT_NONE
   * @see IBk#WEIGHT_INVERSE
   * @see IBk#WEIGHT_SIMILARITY
   */
  public SelectedTag getDistanceWeighting() {
    return m_Classifier.getDistanceWeighting();
  }
  
  /**
   * Sets the distance weighting method used. Values other than
   * WEIGHT_NONE, WEIGHT_INVERSE, or WEIGHT_SIMILARITY will be ignored.
   *
   * @param newMethod the distance weighting method to use
   * @see IBk#WEIGHT_NONE
   * @see IBk#WEIGHT_INVERSE
   * @see IBk#WEIGHT_SIMILARITY
   */
  public void setDistanceWeighting(SelectedTag newMethod) {
    m_Classifier.setDistanceWeighting(newMethod);
  }

  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String nearestNeighbourSearchAlgorithmTipText() {
    return m_Classifier.nearestNeighbourSearchAlgorithmTipText();
  }
  
  /**
   * Returns the current nearestNeighbourSearch algorithm in use.
   * 
   * @return 		the NearestNeighbourSearch algorithm currently in use.
   */
  public NearestNeighbourSearch getNearestNeighbourSearchAlgorithm() {
    return m_Classifier.getNearestNeighbourSearchAlgorithm();
  }
  
  /**
   * Sets the nearestNeighbourSearch algorithm to be used for finding nearest
   * neighbour(s).
   * 
   * @param value 	The NearestNeighbourSearch class.
   */
  public void setNearestNeighbourSearchAlgorithm(NearestNeighbourSearch value) {
    m_Classifier.setNearestNeighbourSearchAlgorithm(value);
  }

  /**
   * Returns the tip text for this property
   * 
   * @return 		tip text for this property suitable for
   * 			displaying in the explorer/experimenter gui
   */
  public String useNaiveSearchTipText() {
    return "Whether to use the naive list search or the KDTree in finding the neighbors.";
  }
  
  /**
   * Sets whether to use the naive list search (= TRUE) or the KDTree (= FALSE)
   * for finding the neighbors
   * 
   * @param value	if true uses the naive list search, otherwise
   * 			KDTree
   */
  public void setUseNaiveSearch(boolean value) {
    m_UseNaiveSearch = value;
  }

  /**
   * Returns whether the naive search or the KDTree is used to search for the
   * neighbors
   * 
   * @return		true if naive list search is used
   */
  public boolean getUseNaiveSearch() {
    return m_UseNaiveSearch;
  }
  
  /**
   * Returns an enumeration describing the available options.
   *
   * @return 		an enumeration of all the available options.
   */
  @Override
  public Enumeration listOptions() {
    Vector        result;
    Enumeration   en;
    
    result = new Vector();
    
    // ancestor
    en = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());
    
    result.addElement(new Option(
        "\tUses a sorted list (ordered according to distance) instead of the\n"
        + "\tKDTree for finding the neighbors.\n"
        + "\t(default is KDTree)",
        "naive", 0, "-naive"));
    
    // IBk
    en = m_Classifier.listOptions();
    while (en.hasMoreElements()) {
      Option o = (Option) en.nextElement();
      // remove -X, -W and -E
      if (    !o.name().equals("X") 
           && !o.name().equals("W") 
           && !o.name().equals("E") )
        result.addElement(o);
    }

    return result.elements();
  }
  
  /**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
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
   * <pre> -insight
   *  Whether to use the labels of the original test set for more
   *  statistics (not used for learning!).
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
   * <pre> -naive
   *  Uses a sorted list (ordered according to distance) instead of the
   *  KDTree for finding the neighbors.
   *  (default is KDTree)</pre>
   * 
   * <pre> -I
   *  Weight neighbours by the inverse of their distance
   *  (use when k &gt; 1)</pre>
   * 
   * <pre> -F
   *  Weight neighbours by 1 - their distance
   *  (use when k &gt; 1)</pre>
   * 
   * <pre> -K &lt;number of neighbors&gt;
   *  Number of nearest neighbours (k) used in classification.
   *  (Default = 1)</pre>
   * 
   * <pre> -A
   *  The nearest neighbour search algorithm to use (default: LinearNN).
   * </pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    super.setOptions(options);

    setUseNaiveSearch(Utils.getFlag("naive", options));

    m_Classifier.setOptions(options);
    m_KNN = m_Classifier.getKNN();         // backup KNN
    m_Classifier.setCrossValidate(true);   // always on!
    m_Classifier.setWindowSize(0);         // always off!
    m_Classifier.setMeanSquared(false);    // always off!
  }
  
  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;
    
    result  = new Vector();
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    options = m_Classifier.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    if (getUseNaiveSearch())
      result.add("-naive");

    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * Returns the value of the named measure
   * @param measureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  @Override
  public double getMeasure(String measureName) {
    if (measureName.equalsIgnoreCase("measureDeterminedKNN"))
      return m_KNNdetermined;
    else
      return super.getMeasure(measureName);
  }

  /**
   * generates copies of the original datasets and also builds a relation
   * (hashtable) between each original instance and new instance. This is
   * necessary to retrieve the determined class value in the 
   * <code>getDistribution(Instance)</code> method.
   * 
   * @see   #getDistribution(Instance)
   * @throws Exception if anything goes wrong
   */
  @Override
  protected void generateSets() throws Exception {
    super.generateSets();

    m_TrainsetNew = new Instances(m_Trainset);
    m_TestsetNew  = new Instances(m_Testset);
  }
  
  /**
   * internal function for determining the class distribution for an instance, 
   * will be overridden by derived classes. <br/>
   * Here we just retrieve the class value for the given instance that was
   * calculated during building our classifier. It just returns "1" for the
   * class that was predicted, no real distribution.
   * Note: the ReplaceMissingValues filter is applied to a copy of the given
   * instance.
   * 
   * @param instance	the instance to get the distribution for
   * @return		the distribution for the given instance
   * @see 		Classifier#distributionForInstance(Instance)
   * @see 		ReplaceMissingValues
   * @throws Exception	if something goes wrong
   */
  @Override
  protected double[] getDistribution(Instance instance) throws Exception {
    double[]        result;
    int             i;                    
    
    result = new double[instance.numClasses()];
    for (i = 0; i < result.length; i++)
      result[i] = 0.0;
    
    i = Arrays.binarySearch(m_LabeledTestset, instance, new InstanceComparator(false));
    if (i >= 0) {
      result[(int) m_LabeledTestset[i].classValue()] = 1.0;
    }
    else {
      CollectiveHelper.writeToTempFile("train.txt", m_TrainsetNew.toString());
      CollectiveHelper.writeToTempFile("test.txt",  m_TestsetNew.toString());
      throw new Exception("Cannot find test instance: " + instance
        + "\n -> pos=" + i + " = " + m_LabeledTestset[StrictMath.abs(i)]);
    }
    
    return result;
  }

  /**
   * determines the "K" for the neighbors from the training set, 
   * initializes the labels of the test set to "missing" and
   * generates the neighbors for all instances in the test set
   * @throws Exception if initialization fails
   */
  protected void initialize() throws Exception {
    int         i;    
    double      timeStart; 
    double      timeEnd;
    Instances   trainingNew;
    Instances   testNew;

    // determine K
    if (getVerbose())
      System.out.println("\nOriginal KNN = " + m_KNN);
    m_Classifier.setKNN(m_KNN);
    m_Classifier.setCrossValidate(true);
    m_Classifier.buildClassifier(m_TrainsetNew);
    m_Classifier.toString();   // necessary to crossvalidate IBk!
    m_Classifier.setCrossValidate(false);
    m_KNNdetermined = m_Classifier.getKNN();
    if (getVerbose())
      System.out.println("Determined KNN = " + m_KNNdetermined);
    
    // set class labels in test set to "missing"
    for (i = 0; i < m_TestsetNew.numInstances(); i++)
      m_TestsetNew.instance(i).setClassMissing();
    
    // copy data
    trainingNew = new Instances(m_TrainsetNew);
    testNew     = new Instances(m_TestsetNew);
    
    // filter data
    m_Missing.setInputFormat(trainingNew);
    trainingNew = Filter.useFilter(trainingNew, m_Missing);
    testNew     = Filter.useFilter(testNew,     m_Missing);
    
    // create the list of neighbors for the instances in the test set
    m_NeighborsTestset = new Neighbors[m_TestsetNew.numInstances()];
    timeStart = System.currentTimeMillis();
    for (i = 0; i < testNew.numInstances(); i++) {
      m_NeighborsTestset[i] = new Neighbors(testNew.instance(i), m_TestsetNew.instance(i), m_KNNdetermined, trainingNew, testNew);
      m_NeighborsTestset[i].setVerbose(getVerbose() || getDebug());
      m_NeighborsTestset[i].setUseNaiveSearch(getUseNaiveSearch());
      m_NeighborsTestset[i].find();
    }
    timeEnd = System.currentTimeMillis();
    
    if (getVerbose())
      System.out.println("Time for finding neighbors: "
        + Utils.doubleToString((timeEnd - timeStart) / 1000.0, 3));
  }

  /**
   * invalidates the ranks, class values of all neighborhoods 
   */
  protected void invalidate() {
    int         i;
    
    // invalidate ranks and class values
    for (i = 0; i < m_NeighborsTestset.length; i++)
      m_NeighborsTestset[i].invalidate();
  }
  
  /**
   * invalidates the neighbor-structures to re-calculate rank and class values
   * @throws Exception if flipping fails
   */
  protected void update() throws Exception {
    int         i;
    int         start;

    invalidate();
    
    // sort according to rank (rank is automatically calculated!)
    Arrays.sort(m_NeighborsTestset);
    
    // get index of neighbor with missing class
    start = -1;
    for (i = 0; i < m_NeighborsTestset.length; i++) {
      if (m_NeighborsTestset[i].getInstance().classIsMissing()) {
        start = i;
        break;
      }
    }
    
    // update the class label value for the ones with the highest rank
    if (start > -1) {
      i = start;
      while (m_NeighborsTestset[i].getRank() == m_NeighborsTestset[start].getRank()) {
        m_NeighborsTestset[i].updateClassValue();
        i++;
        if (i == m_NeighborsTestset.length)
          break;
      }
    }
  }
  
  /**
   * returns TRUE if all neighborhoods have only labeled neighbors 
   * 
   * @return 		true if all neighboorhoods are completely labeled
   * @throws Exception	if something goes wrong
   */
  protected boolean isFinished() throws Exception {
    boolean       result;
    int           i;
    boolean       updated;
    
    result  = true;
    updated = false;
    
    for (i = 0; i < m_NeighborsTestset.length; i++) {
      if (m_NeighborsTestset[i].getInstance().classIsMissing())
        result = false;
      
      if (m_NeighborsTestset[i].isUpdated())
        updated = true;
    }
    
    if (!result && !updated) {
      System.out.println("Couldn't update!");
      throw new Exception("Couldn't update!");
    }
    
    return result;
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = new Capabilities(this);

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.BINARY_CLASS);
    
    return result;
  }
  
  /**
   * performs the actual building of the classifier
   * @throws Exception if building fails
   */
  @Override
  protected void buildClassifier() throws Exception {
    int       i;
    
    i = 0;
    do {
      i++;
      if (getVerbose())
        System.out.println("Iteration " + i);
      
      update();
    }
    while (!isFinished());
    
    // move instances to sorted array
    m_LabeledTestset = new Instance[m_NeighborsTestset.length];
    for (i = 0; i < m_NeighborsTestset.length; i++)
      m_LabeledTestset[i] = m_NeighborsTestset[i].getInstance();
    Arrays.sort(m_LabeledTestset, new InstanceComparator(false));
  }

  /**
   * initializes the classifier and then calls <code>buildClassifier()</code>
   * 
   * @see		#initialize()
   * @see		#buildClassifier()
   * @throws Exception	if something goes wrong
   */
  @Override
  protected void build() throws Exception {
    initialize();
    buildClassifier();
  }
  
  /**
   * returns some information about the parameters
   * 
   * @return		information about the parameters
   */
  @Override
  protected String toStringParameters() {
    String     result;
    
    if (checkBuiltStatus()) {
      result  = "Used K................: " + m_KNNdetermined + "\n";
      result += "\n";
    }
    else {
      result = "";
    }
    
    return result;
  }
  
  /**
   * returns the best model as string representation. derived classes have to 
   * add additional information here, like printing the classifier etc.
   * 
   * @return		the string representation of the best model
   */
  @Override
  protected String toStringModel() {
    StringBuffer     text;
    
    text = new StringBuffer();
    text.append(super.toStringModel());
    text.append("\n");
    text.append(m_Classifier.toString());
    
    return text.toString();
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 2019 $");
  }
  
  /**
   * Main method for testing this class.
   *
   * @param args the options
   */
  public static void main(String[] args) {
    CollectiveEvaluationUtils.runClassifier(new CollectiveIBk(), args);
  }
}
