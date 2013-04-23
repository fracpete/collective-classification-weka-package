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
 * CollectiveWoods.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.trees;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.classifiers.trees.RandomTree;

/**
 <!-- globalinfo-start -->
 * This collective Classifier uses CollectoveTrees to build predictions on the test set. It divides the test set into folds and successively adds the test instances with the best predictions to the training set.<br/>
 * The first iteration trains solely on the training set and determines the distributions for all the instances in the test set. From these predictions the best are chosen (this number is the same as the number of instances in a fold).<br/>
 * From then on, the classifier is trained with the training file from the previous run plus the determined best instances during the previous iteration.<br/>
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
 * <pre> -I &lt;number of trees&gt;
 *  Number of trees to build.</pre>
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
 * <pre> -F &lt;num&gt;
 *  Number of fold for incremental building.
 *  (default is 2)</pre>
 * 
 * <pre> -cut-off &lt;num&gt;
 *  The number of folds after which the learning process
 *  is stopped - 0 means all folds are executed.
 *  (default is 0)</pre>
 * 
 * <pre> -B
 *  Whether to use bagging instead.</pre>
 * 
 * <pre> -bag-size
 *  The size of the bag, where 0 stands for the 
 *  complete training set, less than zero for a 
 *  percentage of instances (from &lt;0 to -1) and 
 *  greater than zero an absolute number of instances.
 *  (default is 0)</pre>
 * 
 <!-- options-end -->
 *
 * @author Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 * @see RandomTree
 */
public class CollectiveWoods
  extends CollectiveForest {

  /** for serialization */
  private static final long serialVersionUID = -2640291845055500872L;

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  @Override
  public String globalInfo() {
    return   "This collective Classifier uses CollectoveTrees to build "
           + "predictions on the test set. It divides the test set "
           + "into folds and successively adds the test instances "
           + "with the best predictions to the training set.\n"
           + "The first iteration trains solely on the training set "
           + "and determines the distributions for all the instances "
           + "in the test set. From these predictions the best are "
           + "chosen (this number is the same as the number of "
           + "instances in a fold).\n"
           + "From then on, the classifier is trained with the "
           + "training file from the previous run plus the determined "
           + "best instances during the previous iteration.\n";
  }

  /**
   * creates a new instance of the classifier, sets the parameters and
   * trains the classifier
   * 
   * @param nextSeed	the next seed value for the classifier
   * @return		the trained classifier
   * @throws Exception	if something goes wrong
   */
  @Override
  protected Classifier initClassifier(int nextSeed) throws Exception {
    CollectiveTree        tree;

    tree = new CollectiveTree();
    tree.setNumFeatures(m_KValue);
    tree.setMinNum(getMinNum());
    tree.setSeed(nextSeed);
    tree.setTestSet(m_TestsetNew);
    tree.buildClassifier(m_TrainsetNew);

    return tree;
  }
  
  /**
   * Main method for testing this class.
   *
   * @param args the options
   */
  public static void main(String[] args) {
    CollectiveEvaluationUtils.runClassifier(new CollectiveWoods(), args);
  }
}
