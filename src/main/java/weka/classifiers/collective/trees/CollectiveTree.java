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
 * CollectiveTree.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.trees;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.collective.CollectiveRandomizableClassifier;
import weka.classifiers.collective.trees.model.CollectiveTreeModel;
import weka.classifiers.collective.trees.model.CollectiveTreeNode;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 <!-- globalinfo-start -->
 * The CollectiveTree classifier works similar to the RandomTree.But it differs in some ways:<br/>
 * - Binary splits:<br/>
 *   ~ it tries to find attributes that split the data in two,<br/>
 *     roughly equal sized, parts.<br/>
 *   ~ in case of nominal attributes it groups those values<br/>
 *     together that split the training and test instances<br/>
 *     roughly in two<br/>
 * - Stops growing if<br/>
 *   ~ only training instances would be covered<br/>
 *   ~ only test instances in the leaf<br/>
 *   ~ only training instances of one class in a leaf<br/>
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
 * <pre> -K &lt;number of attributes&gt;
 *  Number of attributes to randomly investigate
 *  (&lt;1 = int(log(#attributes)+1)).
 *  (default is 0)</pre>
 *
 * <pre> -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default is 1)</pre>
 *
 * <pre> -debug &lt;level&gt;
 *  Set the debug level, i.e., the amount of debug information printed to stdout (the higher, the more information).
 *  (default is 0, i.e., off - overrides '-D')</pre>
 *
 <!-- options-end -->
 *
 * @author Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 * @see weka.classifiers.trees.RandomTree
 */
public class CollectiveTree
  extends CollectiveRandomizableClassifier {

  /** for serialization */
  private static final long serialVersionUID = 4665257328177448582L;

  /** Number of features to consider in random feature selection.
      If less than 1 will use int(logM+1) ) */
  protected int m_NumFeatures = 0;

  /** The actual number of features used in the random feature selection. */
  protected int m_KValue = 0;

  /** Minimum number of instances for leaf. */
  protected double m_MinNum = 1.0;

  /** The random generator for distributing attributes if they have a
   * missing value. */
  protected Random m_RandomAtt = null;

  /** The random generator for distributing instances if they have a
   * missing class. */
  protected Random m_RandomClass = null;

  /** Debug level. */
  protected int m_DebugLevel = 0;

  /** The decision tree model. */
  protected CollectiveTreeModel m_TreeModel = null;

  /** The number of pure training nodes that were prevented */
  protected int m_PureTrainNodes = 0;

  /** The number of pure test nodes that were prevented */
  protected int m_PureTestNodes = 0;

  /**
   * performs initialization of members
   */
  @Override
  protected void initializeMembers() {
    super.initializeMembers();

    m_NumFeatures    = 0;
    m_KValue         = 0;
    m_MinNum         = 1.0;
    m_RandomAtt      = null;
    m_RandomClass    = null;
    m_DebugLevel     = 0;
    m_TreeModel      = null;
    m_PureTrainNodes = 0;
    m_PureTestNodes  = 0;

    m_AdditionalMeasures.add("measureKValue");
    m_AdditionalMeasures.add("measurePureNodes");
    m_AdditionalMeasures.add("measurePureTrainNodes");
    m_AdditionalMeasures.add("measurePureTestNodes");
  }

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return   "The CollectiveTree classifier works similar to the RandomTree."
           + "But it differs in some ways:\n"
           + "- Binary splits:\n"
           + "  ~ it tries to find attributes that split the data in two,\n"
           + "    roughly equal sized, parts.\n"
           + "  ~ in case of nominal attributes it groups those values\n"
           + "    together that split the training and test instances\n"
           + "    roughly in two\n"
           + "- Stops growing if\n"
           + "  ~ only training instances would be covered\n"
           + "  ~ only test instances in the leaf\n"
           + "  ~ only training instances of one class in a leaf\n";
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numFeaturesTipText() {
    return "The number of attributes to be used in random selection (see RandomTree).";
  }

  /**
   * Get the number of features used in random selection.
   *
   * @return Value of numFeatures.
   */
  public int getNumFeatures() {
    return m_NumFeatures;
  }

  /**
   * Set the number of features to use in random selection.
   *
   * @param newNumFeatures Value to assign to numFeatures.
   */
  public void setNumFeatures(int newNumFeatures) {
    m_NumFeatures = newNumFeatures;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String minNumTipText() {
    return "The minimum total weight of the instances in a leaf.";
  }

  /**
   * Get the value of MinNum.
   *
   * @return Value of MinNum.
   */
  public double getMinNum() {
    return m_MinNum;
  }

  /**
   * Set the value of MinNum.
   *
   * @param newMinNum Value to assign to MinNum.
   */
  public void setMinNum(double newMinNum) {
    if (newMinNum >= 1.0)
      m_MinNum = newMinNum;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String debugLevelTipText() {
    return "The debug level (the higher the number the more output, 0 = off).";
  }

  /**
   * Returns the current debug level (0 = off).
   *
   * @return		the debug level
   */
  public int getDebugLevel() {
    return m_DebugLevel;
  }

  /**
   * Sets the new debug level (0 = off), i.e., the higher the level the more
   * printout on stdout.
   *
   * @param newDebugLevel	the new debug level
   */
  public void setDebugLevel(int newDebugLevel) {
    if (newDebugLevel >= 0) {
      m_DebugLevel = newDebugLevel;
      m_Debug      = (m_DebugLevel > 0);
    }
  }

  /**
   * Set debugging mode.
   *
   * @param debug true if debug output should be printed
   */
  @Override
  public void setDebug(boolean debug) {
    super.setDebug(debug);

    if (!debug)
      m_DebugLevel = 0;
    else
      m_DebugLevel = 1;
  }

  /**
   * Lists the command-line options for this classifier.
   *
   * @return		all available options
   */
  @Override
  public Enumeration listOptions() {
    Vector        result;
    Enumeration   en;

    result = new Vector();
    en     = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());

    result.addElement(
      new Option(  "\tNumber of attributes to randomly investigate\n"
                 +"\t(<1 = int(log(#attributes)+1)).\n"
                 + "\t(default is 0)",
                 "K", 1, "-K <number of attributes>"));

    result.addElement(
      new Option(  "\tSet minimum number of instances per leaf.\n"
                 + "\t(default is 1)",
                 "M", 1, "-M <minimum number of instances>"));

    result.addElement(
      new Option(  "\tSet the debug level, i.e., the amount of debug information"
                 + "\tprinted to stdout (the higher, the more information).\n"
                 + "\t(default is 0, i.e., off - overrides '-D')",
                 "debug", 1, "-debug <level>"));

    return result.elements();
  }

  /**
   * Gets options from this classifier.
   *
   * @return		the current setup
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

    result.add("-K");
    result.add("" + getNumFeatures());

    result.add("-M");
    result.add("" + getMinNum());

    if (getDebugLevel() > 0) {
      result.add("-debug");
      result.add("" + getDebugLevel());
    }

    return (String[]) result.toArray(new String[result.size()]);
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
   * <pre> -K &lt;number of attributes&gt;
   *  Number of attributes to randomly investigate
   *  (&lt;1 = int(log(#attributes)+1)).
   *  (default is 0)</pre>
   *
   * <pre> -M &lt;minimum number of instances&gt;
   *  Set minimum number of instances per leaf.
   *  (default is 1)</pre>
   *
   * <pre> -debug &lt;level&gt;
   *  Set the debug level, i.e., the amount of debug information printed to stdout (the higher, the more information).
   *  (default is 0, i.e., off - overrides '-D')</pre>
   *
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    String        tmpStr;

    super.setOptions(options);

    tmpStr = Utils.getOption('K', options);
    if (tmpStr.length() != 0)
      setNumFeatures(Integer.parseInt(tmpStr));
    else
      setNumFeatures(0);

    tmpStr = Utils.getOption('M', options);
    if (tmpStr.length() != 0)
      setMinNum(Double.parseDouble(tmpStr));
    else
      setMinNum(1);

    tmpStr = Utils.getOption("debug", options);
    if (tmpStr.length() != 0)
      setDebugLevel(Integer.parseInt(tmpStr));
    else
      setDebugLevel(getDebugLevel());   // in case setDebug() is TRUE!
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
    result.enable(Capability.MISSING_CLASS_VALUES);

    return result;
  }

  /**
   * Returns the value of the named measure
   * @param measureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  @Override
  public double getMeasure(String measureName) {
    if (measureName.equalsIgnoreCase("measureKValue"))
      return m_KValue;
    else if (measureName.equalsIgnoreCase("measurePureNodes"))
      return m_PureTestNodes + m_PureTrainNodes;
    else if (measureName.equalsIgnoreCase("measurePureTrainNodes"))
      return m_PureTrainNodes;
    else if (measureName.equalsIgnoreCase("measurePureTestNodes"))
      return m_PureTestNodes;
    else
      return super.getMeasure(measureName);
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
  protected double[] getDistribution(Instance instance)
    throws Exception {

    return m_TreeModel.distributionForInstance(instance);
  }

  /**
   * Checks the data, whether it can be used. If not Exceptions are thrown
   * @throws Exception if the data doesn't fit in any way
   */
  @Override
  protected void checkData() throws Exception {
    super.checkData();

    if (m_Trainset.numAttributes() == 1)
      throw new Exception("Data contains only class attribute!");
  }

  /**
   * performs the actual building of the classifier
   *
   * @throws Exception if building fails
   */
  @Override
  protected void buildClassifier() throws Exception {
    Instances data = null;
    int[][] sortedIndices = null;
    double[] vals = null;
    int[][] sortedIndicesTrain = null;
    double[] valsTrain = null;
    int[][] sortedIndicesTest = null;
    double[] valsTest = null;

    m_PureTrainNodes = 0;
    m_PureTestNodes  = 0;

    // Create array of sorted indices (0 = Train, 1 = Test)
    for (int n = 0; n < 2; n++) {
      if (n == 0) {
        data = m_Trainset;

        sortedIndicesTrain = new int[data.numAttributes()][0];
        valsTrain          = new double[data.numInstances()];

        sortedIndices = sortedIndicesTrain;
        vals          = valsTrain;
      }
      else {
        data = m_Testset;

        sortedIndicesTest = new int[data.numAttributes()][0];
        valsTest          = new double[data.numInstances()];

        sortedIndices = sortedIndicesTest;
        vals          = valsTest;
      }

      for (int j = 0; j < data.numAttributes(); j++) {
        if (j != data.classIndex()) {
          if (data.attribute(j).isNominal()) {
            // Handling nominal attributes. Putting indices of
            // instances with missing values at the end.
            sortedIndices[j] = new int[data.numInstances()];
            int count = 0;
            for (int i = 0; i < data.numInstances(); i++) {
              Instance inst = data.instance(i);
              if (!inst.isMissing(j)) {
                sortedIndices[j][count] = i;
                count++;
              }
            }
            for (int i = 0; i < data.numInstances(); i++) {
              Instance inst = data.instance(i);
              if (inst.isMissing(j)) {
                sortedIndices[j][count] = i;
                count++;
              }
            }
          }
          else {
            // Sorted indices are computed for numeric attributes
            for (int i = 0; i < data.numInstances(); i++) {
              Instance inst = data.instance(i);
              vals[i] = inst.value(j);
            }
            sortedIndices[j] = Utils.sort(vals);
          }
        }
      }
    }

    // Compute initial class counts
    double[] classProbs = new double[m_Trainset.numClasses()];
    for (int i = 0; i < m_Trainset.numInstances(); i++) {
      Instance inst = m_Trainset.instance(i);
      classProbs[(int)inst.classValue()] += inst.weight();
    }

    // Create the attribute indices window
    int[] attIndicesWindow = new int[m_Trainset.numAttributes()-1];
    int j=0;
    for (int i=0; i<attIndicesWindow.length; i++) {
      if (j == m_Trainset.classIndex()) j++; // do not include the class
      attIndicesWindow[i] = j++;
    }

    // create random generator for distributing attributes if they have a
    // missing value
    m_RandomAtt = m_Trainset.getRandomNumberGenerator(getSeed());

    // create random generator for distributing instances if they have a
    // missing class
    m_RandomClass = m_Trainset.getRandomNumberGenerator(getSeed());

    // Build tree
    buildTree(null,
              sortedIndicesTrain,
              sortedIndicesTest,
              classProbs,
	      attIndicesWindow, m_Trainset.getRandomNumberGenerator(getSeed()));

    if (getDebugLevel() > 1)
      System.out.println(m_TreeModel.getString());
  }

  /**
   * Recursively generates a tree.
   *
   * @param node		the parent node
   * @param sortedIndicesTrain	the sorted indices of the training set
   * @param sortedIndicesTest	the sorted indices of the test set
   * @param classProbs		the class probabilities
   * @param attIndicesWindow	the currently available attributes
   * @param random		the random number generator
   * @throws Exception		if something goes wrong
   */
  protected void buildTree(CollectiveTreeNode node,
                           int[][] sortedIndicesTrain,
                           int[][] sortedIndicesTest,
                           double[] classProbs,
			   int[] attIndicesWindow, Random random)
    throws Exception {

    double[][]          newDistribution;
    double[]            newClassProbs;
    int                 newAttribute;
    double              newSplitPoint;
    double[][]          newNominalSplit;
    double[]            newProp;
    CollectiveTreeNode  child;
    CollectiveTreeNode  parent;

    if (getDebugLevel() > 0)
      System.out.println("attIndicesWindow: "
                          + Utils.arrayToString(attIndicesWindow));

    // root?
    if (node == null) {
      node = new CollectiveTreeNode(null);
      node.setInformation(new Instances(m_Trainset, 0));
      m_TreeModel = new CollectiveTreeModel(node);
      m_TreeModel.setTreeName("CollectiveTree");
      m_TreeModel.getRootNode().setClassProbabilities(m_Trainset);
      //m_TreeModel.setDebugLevel(getDebugLevel());
    }

    // determine parent
    parent = (CollectiveTreeNode) node.getParent();

    // Make leaf if there are no training instances
    if (((m_Trainset.classIndex()  > 0) && (sortedIndicesTrain[0].length == 0)) ||
	((m_Trainset.classIndex() == 0) && (sortedIndicesTrain[1].length == 0))) {
      newDistribution = new double[1][m_Trainset.numClasses()];
      newClassProbs = null;

      node.setDistribution(newDistribution);
      node.setClassProbabilities(newClassProbs);

      if (getDebugLevel() > 0)
        System.out.println("\nNo training instances left\n" + node.toStringNode());

      m_PureTestNodes++;

      return;
    }

    // Make leaf if there are no test instances -> distr. from parent
    if (((m_Testset.classIndex()  > 0) && (sortedIndicesTest[0].length == 0)) ||
	((m_Testset.classIndex() == 0) && (sortedIndicesTest[1].length == 0))) {

      node.setDistribution(parent.getDistribution());
      node.setClassProbabilities(parent.getClassProbabilities());

      if (getDebugLevel() > 0)
        System.out.println("\nNo test instances left\n" + node.toStringNode());

      m_PureTrainNodes++;

      return;
    }

    // Check if node doesn't contain enough instances or is pure
    newClassProbs = new double[classProbs.length];
    System.arraycopy(classProbs, 0, newClassProbs, 0, classProbs.length);
    if (   Utils.sm(Utils.sum(newClassProbs), 2 * m_MinNum)
	|| Utils.eq(newClassProbs[Utils.maxIndex(newClassProbs)],
		 Utils.sum(newClassProbs)) ) {

      // Make leaf
      newAttribute = -1;
      newDistribution = new double[1][newClassProbs.length];
      for (int i = 0; i < newClassProbs.length; i++) {
	newDistribution[0][i] = newClassProbs[i];
      }
      Utils.normalize(newClassProbs);

      node.setClassProbabilities(newClassProbs);
      node.setAttribute(newAttribute);
      node.setDistribution(newDistribution);

      if (getDebugLevel() > 0)
        System.out.println("\nToo small or pure\n" + node.toStringNode());

      return;
    }

    // Compute class distributions and value of splitting
    // criterion for each attribute
    double[] vals = new double[m_Trainset.numAttributes()];
    double[][][] dists = new double[m_Trainset.numAttributes()][0][0];
    double[][][] distsTest = new double[m_Trainset.numAttributes()][0][0];
    double[][] props = new double[m_Trainset.numAttributes()][0];
    double[][] propsTest = new double[m_Trainset.numAttributes()][0];
    double[] splits = new double[m_Trainset.numAttributes()];
    double[][][] nominalSplits = new double[m_Trainset.numAttributes()][2][];

    // Investigate K random attributes
    int attIndex = 0;
    int windowSize = attIndicesWindow.length;
    int k = m_KValue;
    boolean gainFound = false;
    while ((windowSize > 0) && (k-- > 0 || !gainFound)) {

      int chosenIndex = random.nextInt(windowSize);
      attIndex = attIndicesWindow[chosenIndex];
      if (getDebugLevel() > 1)
        System.out.println(  "k=" + k + " -> chosenIndex=" + chosenIndex
                           + " -> attIndex=" + attIndex);

      // shift chosen attIndex out of window
      attIndicesWindow[chosenIndex]  = attIndicesWindow[windowSize-1];
      attIndicesWindow[windowSize-1] = attIndex;
      windowSize--;

      // determine split data (point or nominalsplit)
      splits[attIndex] = determineSplit( sortedIndicesTrain[attIndex],
                                         sortedIndicesTest[attIndex],
                                         attIndex, nominalSplits );

      distribution(props, dists, attIndex,
                   splits[attIndex], nominalSplits[attIndex],
	           sortedIndicesTrain[attIndex],
                   sortedIndicesTest[attIndex], true);
      distribution(propsTest, distsTest, attIndex,
                   splits[attIndex], nominalSplits[attIndex],
	           sortedIndicesTrain[attIndex],
                   sortedIndicesTest[attIndex], false);

      vals[attIndex] = gain(dists[attIndex], priorVal(dists[attIndex]));

      if (Utils.gr(vals[attIndex], 0))
        gainFound = true;
      if (getDebugLevel() > 1)
        System.out.println("vals=" + vals[attIndex] + " -> gain=" + gainFound);
    }

    // Find best attribute
    newAttribute = Utils.maxIndex(vals);
    newDistribution = dists[newAttribute];

    node.setAttribute(newAttribute);
    node.setDistribution(newDistribution);

    // Any useful split found?
    if (Utils.gr(vals[newAttribute], 0)) {

      // Build subtrees
      newSplitPoint   = splits[newAttribute];
      newProp         = props[newAttribute];
      newNominalSplit = nominalSplits[newAttribute];

      node.setSplitPoint(newSplitPoint);
      node.setProportions(newProp);
      node.setNominalSplit(newNominalSplit);

      if (getDebugLevel() > 0)
        System.out.println("\nUseful split\n" + node.toStringNode());

      // training
      int[][][] subsetIndicesTrain =
        new int[newDistribution.length][m_Trainset.numAttributes()][0];

      splitData(node, subsetIndicesTrain,
                newAttribute, newSplitPoint, newNominalSplit,
                sortedIndicesTrain, newDistribution, m_Trainset);

      // test
      int[][][] subsetIndicesTest =
        new int[newDistribution.length][m_Testset.numAttributes()][0];

      splitData(node, subsetIndicesTest,
                newAttribute, newSplitPoint, newNominalSplit,
                sortedIndicesTest, newDistribution, m_Testset);

      for (int i = 0; i < newDistribution.length; i++) {
        child = new CollectiveTreeNode(node);
	buildTree(child,
                  subsetIndicesTrain[i],
                  subsetIndicesTest[i],
		  newDistribution[i],
		  attIndicesWindow, random);
      }
    }
    else {

      // Make leaf
      newAttribute = -1;
      newDistribution = new double[1][newClassProbs.length];
      for (int i = 0; i < newClassProbs.length; i++) {
	newDistribution[0][i] = newClassProbs[i];
      }

      node.setAttribute(newAttribute);
      node.setDistribution(newDistribution);

      if (getDebugLevel() > 0)
        System.out.println("\nNo useful split\n" + node.toStringNode());
    }

    // Normalize class counts
    Utils.normalize(newClassProbs);

    node.setClassProbabilities(newClassProbs);
  }

  /**
   * Returns size of the tree.
   *
   * @return 		the size of the tree
   */
  public int numNodes() {
    if (m_TreeModel == null)
      return 0;
    else
      return m_TreeModel.size();
  }

  /**
   * Splits instances into subsets.
   * @param node              the paren node
   * @param subsetIndices     will be filled with sorted indices
   * @param att               the attribute to work on
   * @param splitPoint        the split point for numeric attributes
   * @param nominalSplit      the split bins (containing att. values) for
   *                          nominal attributes
   * @param sortedIndices     the indices to split
   * @param dist              the class distribution
   * @param data              the instances to work on (linked to the sorted
   *                          indices)
   * @throws Exception        if something goes wrong
   */
  protected void splitData(CollectiveTreeNode node,
                           int[][][] subsetIndices,
			   int att, double splitPoint, double[][] nominalSplit,
			   int[][] sortedIndices,
			   double[][] dist, Instances data) throws Exception {

    int j;
    int[] num;
    double[] newProp = node.getProportions();

    // For each attribute
    for (int i = 0; i < data.numAttributes(); i++) {
      if (i != data.classIndex()) {
	// For nominal attributes
	if (data.attribute(att).isNominal()) {
	  num = new int[nominalSplit.length];

	  for (int k = 0; k < num.length; k++)
	    subsetIndices[k][i] = new int[sortedIndices[i].length];

	  for (j = 0; j < sortedIndices[i].length; j++) {
	    Instance inst = data.instance(sortedIndices[i][j]);
	    // missing -> Split instance up
	    if (inst.isMissing(att)) {
	      for (int k = 0; k < num.length; k++) {
		if (Utils.gr(newProp[k], 0)) {
		  subsetIndices[k][i][num[k]] = sortedIndices[i][j];
		  num[k]++;
		}
	      }
	    }
            else {
              int subset = determineBranch(inst, att, nominalSplit, null)[0];
	      subsetIndices[subset][i][num[subset]] = sortedIndices[i][j];
	      num[subset]++;
	    }
	  }
	}
	// For numeric attributes
        else {
	  num = new int[2];

	  for (int k = 0; k < 2; k++)
	    subsetIndices[k][i] = new int[sortedIndices[i].length];

          for (j = 0; j < sortedIndices[i].length; j++) {
	    Instance inst = data.instance(sortedIndices[i][j]);
	    // missing -> Split instance up
	    if (inst.isMissing(att)) {
	      for (int k = 0; k < num.length; k++) {
		if (Utils.gr(newProp[k], 0)) {
		  subsetIndices[k][i][num[k]] = sortedIndices[i][j];
		  num[k]++;
		}
	      }
	    }
            else {
	      int subset = Utils.sm(inst.value(att), splitPoint) ? 0 : 1;
	      subsetIndices[subset][i][num[subset]] = sortedIndices[i][j];
	      num[subset]++;
	    }
	  }
	}

	// Trim arrays
	for (int k = 0; k < num.length; k++) {
	  int[] copy = new int[num[k]];
	  System.arraycopy(subsetIndices[k][i], 0, copy, 0, num[k]);
	  subsetIndices[k][i] = copy;
	}
      }
    }
  }

  /**
   * determines the split point (if numeric) and the nominal split (if
   * nominal) of a given attribute
   * @param indicesTrain  the sorted indices of the training set
   * @param indicesTest   the sorted indices of the test set
   * @param att           the attribute to split on
   * @param nominalSplits will be filled with the nominal split values, i.e.
   *                      two branches containing each some nominal values
   * @return              the split point if numeric
   */
  protected double determineSplit( int[] indicesTrain,
                                   int[] indicesTest,
                                   int att, double[][][] nominalSplits ) {
    double        splitPoint;
    Attribute     attribute;
    double[][]    nominalSplit;
    double[][]    dist;
    Instance      inst;
    int           i;

    splitPoint   = Double.NaN;
    attribute    = m_Trainset.attribute(att);
    nominalSplit = null;

    if (getDebugLevel() > 2) {
      System.out.println(   "Attribute: " + attribute.name()
                          + " (nominal=" + attribute.isNominal() + ")");
      System.out.println(   "train-size=" + indicesTrain.length
                          + ", test-size=" + indicesTest.length );
    }

    // not enough instances? -> no splitpoint and empty nominal split
    if ( (indicesTrain.length == 0) || (indicesTest.length == 0) ) {
      if (getDebugLevel() > 0)
        System.out.println("Warning: length too small!");

      splitPoint         = Double.NaN;
      nominalSplit       = new double[2][1];
      nominalSplits[att] = nominalSplit;

      return splitPoint;
    }

    // nominal attributes
    if (attribute.isNominal()) {
      dist = new double[attribute.numValues()][m_Trainset.numClasses() + 1];
      // training
      for (i = 0; i < indicesTrain.length; i++) {
	inst = m_Trainset.instance(indicesTrain[i]);
	if (inst.isMissing(att))
	  break;
	dist[(int)inst.value(att)][(int)inst.classValue()] += inst.weight();
      }
      // test -> as extra class label, i.e., 3. label besides pos and neg
      for (i = 0; i < indicesTest.length; i++) {
	inst = m_Testset.instance(indicesTest[i]);
	if (inst.isMissing(att))
	  break;
	dist[(int)inst.value(att)][m_Trainset.numClasses()] += inst.weight();
      }
      if (getDebugLevel() > 2)
        System.out.println("dist: " + Utils.arrayToString(dist));

      // calculate ranking with (pos+1)/(pos+neg+2)
      double[] vals = new double[attribute.numValues()];
      double[] sums = new double[attribute.numValues()];
      for (i = 0; i < attribute.numValues(); i++) {
        vals[i] = (   (dist[i][0] + 1))
                    / (dist[i][0] + dist[i][1] + 2);
        sums[i] = dist[i][0] + dist[i][1] + dist[i][2];
      }
      if (getDebugLevel() > 2) {
        System.out.println("vals: " + Utils.arrayToString(vals));
        System.out.println("sums: " + Utils.arrayToString(sums));
      }

      // get ranking (equally ranked are merged!)
      int[] sorted = Utils.sort(vals);
      Vector merged = new Vector();
      int n = 0;
      double currVal = 0;
      for (i = 0; i < sorted.length; i++) {
        if (i == 0) {
          merged.add(new Vector());
          ((Vector) merged.get(n)).add(new Integer(sorted[i]));
        }
        else {
          if (Utils.eq(vals[sorted[i]], currVal)) {
            ((Vector) merged.get(n)).add(new Integer(sorted[i]));
          }
          else {
            n++;
            merged.add(new Vector());
            ((Vector) merged.get(n)).add(new Integer(sorted[i]));
          }
        }
        currVal = vals[sorted[i]];
      }
      if (getDebugLevel() > 2) {
        System.out.println("sorted: " + Utils.arrayToString(sorted));
        System.out.println("merged: " + merged);
      }

      // sum up merged counts
      double[] mergedSums = new double[merged.size()];
      double totalSum = 0;
      for (i = 0; i < merged.size(); i++) {
        Vector v = (Vector) merged.get(i);
        mergedSums[i] = 0;
        for (n = 0; n < v.size(); n++)
          mergedSums[i] += sums[((Integer) v.get(n)).intValue()];
        totalSum += mergedSums[i];
      }
      if (getDebugLevel() > 2)
        System.out.println("mergedSums: " + Utils.arrayToString(mergedSums));

      // determine best split point (roughly in half)
      // (but always stop before the last element, otherwise the branch could
      // be left empty!)
      currVal  = 0;
      int valIndex = 0;
      int valCount = 0;
      for (i = 0; i < mergedSums.length - 1; i++) {
        // already past half?
        if (currVal >= totalSum / 2) {
          break;
        }
        currVal += mergedSums[i];
        valCount += ((Vector) merged.get(i)).size();
        valIndex = i;
      }

      if (getDebugLevel() > 2) {
        System.out.println("valIndex: " + valIndex);
        System.out.println("valCount: " + valCount);
        System.out.println("numValues: " + attribute.numValues());
      }

      // generate nominal split
      nominalSplit = new double[2][];
      nominalSplit[0] = new double[valIndex + 1];
      nominalSplit[1] = new double[attribute.numValues() - (valIndex + 1)];
      int index = 0;
      for (i = 0; i < merged.size(); i++) {
        Vector v = (Vector) merged.get(i);
        for (n = 0; n < v.size(); n++) {
          // left branch
          if (index < nominalSplit[0].length)
            nominalSplit[0][index] =
              ((Integer) v.get(n)).intValue();
          // right branch
          else
            nominalSplit[1][index - nominalSplit[0].length] =
              ((Integer) v.get(n)).intValue();
          index++;
        }
      }
      if (getDebugLevel() > 2)
        System.out.println("nominalSplit: " + Utils.arrayToString(nominalSplit));
    }
    // For numeric attributes
    else {
      // find split point that separates into two halves
      int indexTrain = indicesTrain.length / 2;
      int indexTest  = indicesTest.length / 2;

      if (getDebugLevel() > 2)
        System.out.println(   "indexTrain=" + indexTrain
                            + ", indexTest=" + indexTest );

      double valueTrain = m_Trainset.instance(indicesTrain[indexTrain]).value(att);
      double valueTest  = m_Testset.instance(indicesTest[indexTest]).value(att);

      if (valueTrain < valueTest) {
        while (    (indexTrain < indicesTrain.length - 1)
                && (indexTest > 0) ) {
          indexTrain++;
          indexTest--;
          valueTrain = m_Trainset.instance(indicesTrain[indexTrain]).value(att);
          valueTest  = m_Testset.instance(indicesTest[indexTest]).value(att);
          // found the threshold?
          if (valueTrain > valueTest) {
            // one step back
            indexTrain--;
            indexTest++;
            break;
          }
        }
      }
      else {
        while (    (indexTrain > 0)
                && (indexTest < indicesTest.length - 1) ) {
          indexTrain--;
          indexTest++;
          valueTrain = m_Trainset.instance(indicesTrain[indexTrain]).value(att);
          valueTest  = m_Testset.instance(indicesTest[indexTest]).value(att);
          // found the threshold?
          if (valueTrain < valueTest) {
            // one step back
            indexTrain++;
            indexTest--;
            break;
          }
        }
      }

      splitPoint = m_Trainset.instance(indicesTrain[indexTrain]).value(att);

      if (getDebugLevel() > 2)
        System.out.println("splitPoint: " + splitPoint);
    }

    // return nominal split and splitpoint
    nominalSplits[att] = nominalSplit;
    return splitPoint;
  }

  /**
   * Computes class distribution for an attribute.
   * @param props           will contain the new proportions
   * @param dists           will contain the new distribution
   * @param att             the attribute to work on
   * @param splitPoint      the split criteria for numeric attributes
   * @param nominalSplit    the split criteria (bins with values) for nominal
   *                        attributes
   * @param indicesTrain    the sorted indices of the training set
   * @param indicesTest     the sorted indices of the test set
   * @param isTrain         whether to work on the training set or not
   * @throws Exception      if something goes wrong
   */
  protected void distribution(double[][] props, double[][][] dists, int att,
                              double splitPoint, double[][] nominalSplit,
                              int[] indicesTrain, int[] indicesTest,
                              boolean isTrain)
    throws Exception {

    // the data to use
    Instances data;
    int[] indices;
    if (isTrain) {
      data    = m_Trainset;
      indices = indicesTrain;
    }
    else {
      data    = m_Testset;
      indices = indicesTest;
    }

    Attribute attribute = data.attribute(att);
    double[][] dist = null;
    int i;
    int[] position;
    int missingIndex = -1;
    double clsValue = Double.NaN;

    // determine distributions
    double[] attDist = determineAttributeDistribution(data, indices, att);
    double[] clsDist = null;
    if (!isTrain)
      clsDist = determineClassDistribution(m_Trainset, indicesTrain);

    // For nominal attributes
    if (attribute.isNominal()) {
      dist = new double[nominalSplit.length][data.numClasses()];

      for (i = 0; i < indices.length; i++) {
	Instance inst = data.instance(indices[i]);
	if (inst.isMissing(att)) {
          missingIndex = i;
	  break;
	}

        // get class label
        if (isTrain)
          clsValue = inst.classValue();
        else
          clsValue = determineClass(inst, clsDist);

        position = determineBranch(inst, att, nominalSplit, attDist);
	dist[position[0]][(int) clsValue] += inst.weight();
      }
    }
    // For numeric attributes
    else {
      dist = new double[2][data.numClasses()];

      for (i = 0; i < indices.length; i++) {
	Instance inst = data.instance(indices[i]);
	if (inst.isMissing(att)) {
          missingIndex = i;
	  break;
	}

        // get class label
        if (isTrain)
          clsValue = inst.classValue();
        else
          clsValue = determineClass(inst, clsDist);

        position = determineBranch(inst, att, splitPoint, attDist);
	dist[position[0]][(int) clsValue] += inst.weight();
      }
    }

    // Compute weights
    props[att] = new double[dist.length];
    for (int k = 0; k < props[att].length; k++) {
      props[att][k] = Utils.sum(dist[k]);
    }
    if (Utils.eq(Utils.sum(props[att]), 0)) {
      for (int k = 0; k < props[att].length; k++) {
	props[att][k] = 1.0 / props[att].length;
      }
    }
    else {
      Utils.normalize(props[att]);
    }

    // Any instances with missing values ?
    if (missingIndex > -1) {
      // Distribute counts
      i = missingIndex;
      while (i < indices.length) {
	Instance inst = data.instance(indices[i]);
	for (int j = 0; j < dist.length; j++) {
          // determine class
          if (isTrain)
	    clsValue = inst.classValue();
          else
            clsValue = determineClass(inst, clsDist);

	  dist[j][(int) clsValue] += props[att][j] * inst.weight();
	}
	i++;
      }
    }

    // Return distribution
    dists[att] = dist;
  }

  /**
   * determines the distribution of the instances with a non-missing value
   * at the given attribute position.
   * @param data        the instances to work on
   * @param indices     the sorted indices
   * @param att         the attribute to determine the distribution for
   * @return            the distribution
   */
  protected double[] determineAttributeDistribution( Instances data,
                                                     int[] indices,
                                                     int att) {
    double[]      result;
    int           i;
    Instance      inst;
    int           count;
    double[]      values;
    double        median;

    // nominal attribute
    if (data.attribute(att).isNominal()) {
      result = new double[data.attribute(att).numValues()];

      // determine attribute distribution (necessary to distribute instances
      // with no class and missing attribute)
      for (i = 0; i < indices.length; i++) {
        inst = data.instance(indices[i]);
        if (inst.isMissing(att))
          break;
        result[(int) inst.value(att)] += inst.weight();
      }
    }
    // numeric attribute
    else {
      result = new double[2];   // less or greater/equal than median

      // determine number of instances w/o missing attribute
      count = 0;
      for (i = 0; i < indices.length; i++) {
        inst = data.instance(indices[i]);
        if (inst.isMissing(att))
          break;
        count++;
      }

      // determine median
      values = new double[count];
      for (i = 0; i < count; i++) {
        inst      = data.instance(indices[i]);
        values[i] = inst.value(att);
      }
      if (values.length == 0)
        median = 0;
      else if (values.length == 1)
        median = values[0];
      else
        median = Utils.kthSmallestValue(values, values.length / 2);

      // disitribute
      for (i = 0; i < count; i++) {
        inst = data.instance(indices[i]);
        if (Utils.sm(inst.value(att), median))
          result[0] += inst.weight();
        else
          result[1] += inst.weight();
      }
    }

    if (Utils.gr(Utils.sum(result), 0))
      Utils.normalize(result);

    return result;
  }

  /**
   * determines the class distribution of the instances with a non-missing
   * value as class value.
   * @param data        the instances to work on
   * @param indices     the sorted indices
   * @return            the distribution
   */
  protected double[] determineClassDistribution(Instances data, int[] indices) {
    double[]      result;
    int           i;
    Instance      inst;

    result = new double[data.numClasses()];

    for (i = 0; i < indices.length; i++) {
      inst = data.instance(indices[i]);
      if (inst.classIsMissing())
        break;
      result[(int) inst.classValue()] += inst.weight();
    }

    return result;
  }

  /**
   * determines the branch and index in that branch the given instance will go
   * into, according to the nominal split
   * @param inst        the instance to distribute into a branch
   * @param att         the attribute index
   * @param split       the nominal split on the given attribute
   * @param attDist     the attribute distribution (of instances with no
   *                    missing value at the attribute's position)
   * @return            the branch (index 0) and the index in this branch
   *                    (index 1), -1 if an error happened
   */
  protected int[] determineBranch( Instance inst, int att,
                                   double[][] split, double[] attDist) {
    int           i;
    int           n;
    int[]         result;
    double        val;
    double        currVal;

    result    = new int[2];
    result[0] = -1;
    result[1] = -1;

    // missing? -> randomly, according to attribute distribution
    if (inst.isMissing(att)) {
      val = m_RandomAtt.nextDouble();
      // determine branch the random number fits into
      currVal = 0;
      for (i = 0; i < attDist.length; i++) {
        if ( (val >= currVal) && (val < attDist[i]) ) {
          result[0] = i;
          result[1] = (int) currVal;
          break;
        }
        currVal = attDist[i];
      }
    }
    // find value in the nominal splits
    else {
      for (i = 0; i < split.length; i++) {
        for (n = 0; n < split[i].length; n++) {
          // found?
          if (Utils.eq(inst.value(att), split[i][n])) {
            result[0] = i;
            result[1] = n;
            break;
          }
        }

        if (result[0] != -1)
          break;
      }
    }

    return result;
  }

  /**
   * determines the branch and the index in that branch (in the numeric case
   * this is always 0) the given instance will go into, according to the
   * numeric split
   * @param inst        the instance to distribute into a branch
   * @param att         the attribute index
   * @param split       the nominal split on the given attribute
   * @param attDist     the attribute distribution (of instances with no
   *                    missing value at the attribute's position)
   * @return            the branch (index 0) and the index in that branch
   *                    (which is in the numeric case always 0), -1 if an
   *                    error happened
   */
  protected int[] determineBranch( Instance inst, int att,
                                   double split, double[] attDist) {
    int           i;
    int[]         result;
    double        val;
    double        currVal;

    result    = new int[2];
    result[0] = -1;

    // missing? -> randomly, according to attribute distribution
    if (inst.isMissing(att)) {
      val = m_RandomAtt.nextDouble();
      // determine branch the random number fits into
      currVal = 0;
      for (i = 0; i < attDist.length; i++) {
        if ( (val >= currVal) && (val < attDist[i]) ) {
          result[0] = i;
          break;
        }
        currVal = attDist[i];
      }
    }
    // find value in the nominal splits
    else {
      if (inst.value(att) < split)
        result[0] = 0;
      else
        result[0] = 1;
    }

    return result;
  }

  /**
   * determines the class of the instance. I.e. if it's not missing it just
   * returns it, otherwise it returns a random class based on the class
   * distribution
   * @param inst        the instance to get the class for
   * @param classDist   the class distribution
   * @return            the class for the instance
   */
  protected double determineClass(Instance inst, double[] classDist) {
    double        result;
    double        val;
    double        currVal;
    int           i;

    result = Utils.missingValue();

    if (inst.classIsMissing()) {
      val = m_RandomClass.nextDouble() * Utils.sum(classDist);
      // determine class the random number fits into
      currVal = 0;
      for (i = 0; i < classDist.length; i++) {
        if ( (val >= currVal) && (val < classDist[i]) ) {
          result = i;
          break;
        }
        currVal = classDist[i];
      }
    }
    else {
      result = inst.classValue();
    }

    return result;
  }

  /**
   * Computes value of splitting criterion before split.
   *
   * @param dist	the distribution
   * @return		prior val
   */
  protected double priorVal(double[][] dist) {
    return ContingencyTables.entropyOverColumns(dist);
  }

  /**
   * Computes value of splitting criterion after split.
   *
   * @param dist	the distribution
   * @param priorVal	the prior val
   * @return		the gain
   */
  protected double gain(double[][] dist, double priorVal) {
    return priorVal - ContingencyTables.entropyConditionedOnRows(dist);
  }

  /**
   * here initialization and building, possible iterations will happen
   *
   * @throws Exception	if something goes wrong
   */
  @Override
  protected void build() throws Exception {
    // determine number of features to be selected
    m_KValue = getNumFeatures();
    if (m_KValue < 1)
      m_KValue = (int) Utils.log2(m_Trainset.numAttributes()) + 1;

    // Make sure K value is in range
    if (m_KValue > m_Trainset.numAttributes() - 1)
      m_KValue = m_Trainset.numAttributes() - 1;

    // build classifier
    m_Random = m_Trainset.getRandomNumberGenerator(getSeed());
    buildClassifier();
  }

  /**
   * returns information about the classifier(s)
   *
   * @return		information about the classifier
   */
  @Override
  protected String toStringClassifier() {
    return "";
  }

  /**
   * returns some information about the parameters
   *
   * @return		information about the parameters
   */
  @Override
  protected String toStringParameters() {
    String        result;

    result =   "Features...............: " + m_KValue + "\n"
             + "Min Number of instances: " + getMinNum() + "\n";

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
    return m_TreeModel.getString();
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
    CollectiveEvaluationUtils.runClassifier(new CollectiveTree(), args);
  }
}
