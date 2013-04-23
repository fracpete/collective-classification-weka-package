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
 * CollectiveNeighbor.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.meta;

import weka.classifiers.evaluation.CollectiveEvaluationUtils;


/**
 <!-- globalinfo-start -->
 * Dummy classifier for TwoStageCollective - only used to keep experiments working!
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
 * <pre> -V
 *  Inverts the fold selection, i.e., instead of using the first
 *  fold for the training set it is used for test set and the
 *  remaining folds for training.</pre>
 * 
 * <pre> -log
 *  Creates logs in the tmp directory for all kinds of internal data.
 *  Use only for debugging purposes!
 * </pre>
 * 
 * <pre> -eval
 *  The type of evaluation to use (0 = Randomwalk/Last model used for 
 *  prediction, 1=Randomwalk/Best model used for prediction,
 *  2=Hillclimbing).
 * </pre>
 * 
 * <pre> -compare
 *  The type of comparisong used for comparing models.
 *  (0=overall RMS, 1=RMS on train set, 2=RMS on test set)
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
 * <pre> -B &lt;classifier specification&gt;
 *  Full class name of classifier to include, followed
 *  by scheme options. May be specified multiple times.
 *  (default: "weka.classifiers.rules.ZeroR")</pre>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * @author Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $ 
 * @see     TwoStageCollective
 */
public class CollectiveNeighbor 
  extends TwoStageCollective {
  
  /** for serialization */
  private static final long serialVersionUID = -4606667554231860447L;

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  @Override
  public String globalInfo() {
    return "Dummy classifier for TwoStageCollective - only used to keep experiments working!";
  }
  
  /**
   * Main method for testing this class.
   *
   * @param args the options
   */
  public static void main(String[] args) {
    CollectiveEvaluationUtils.runClassifier(new CollectiveNeighbor(), args);
  }
}
