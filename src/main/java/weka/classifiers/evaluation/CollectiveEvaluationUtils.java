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
 *    CollectiveEvaluationUtils.java
 *    Copyright (C) 2002-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.evaluation;

import weka.classifiers.Classifier;
import weka.classifiers.CollectiveEvaluation;
import weka.classifiers.collective.CollectiveClassifier;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.RevisionUtils;

/**
 * Contains utility functions for generating lists of predictions in 
 * various manners.
 *
 * @author fracpete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class CollectiveEvaluationUtils
  extends EvaluationUtils {

  /**
   * Generate a bunch of predictions ready for processing, by performing a
   * evaluation on a test set after training on the given training set.
   *
   * @param classifier the Classifier to evaluate
   * @param train the training dataset
   * @param test the test dataset
   * @exception Exception if an error occurs
   */
  @Override
  public FastVector getTrainTestPredictions(Classifier classifier, 
                                            Instances train, Instances test) 
    throws Exception {
    
    if (classifier instanceof CollectiveClassifier) {
      CollectiveClassifier c = (CollectiveClassifier) classifier;
      c.reset();
      c.setTestSet(test);
    }

    return super.getTrainTestPredictions(classifier, train, test);
  }

  /**
   * runs the classifier instance with the given options.
   *
   * @param classifier	the classifier to run
   * @param options	the commandline options
   */
  public static void runClassifier(CollectiveClassifier classifier, String[] options) {
    try {
      System.out.println(CollectiveEvaluation.evaluateModel(classifier, options));
    }
    catch (Exception e) {
      if (    ((e.getMessage() != null) && (e.getMessage().indexOf("General options") == -1))
          || (e.getMessage() == null) )
        e.printStackTrace();
      else
        System.err.println(e.getMessage());
    }
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
}
