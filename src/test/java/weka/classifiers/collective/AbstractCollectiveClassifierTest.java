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
 * Copyright (C) 2005-2015 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.collective;

import java.util.ArrayList;

import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.CheckClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.CollectiveEvaluationUtils;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.classifiers.evaluation.Prediction;
import weka.core.CheckGOE;
import weka.core.Instances;

/**
 * Ancestor collective classifier test classes.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public abstract class AbstractCollectiveClassifierTest 
  extends AbstractClassifierTest {
  
  /**
   * Constructs the <code>AbstractCollectiveClassifierTest</code>. Called by subclasses.
   *
   * @param name the name of the test class
   */
  public AbstractCollectiveClassifierTest(String name) { 
    super(name); 
  }

  @Override
  protected CheckClassifier getTester() {
    CheckCollectiveClassifier result;

    result = new CheckCollectiveClassifier();
    result.setSilent(true);
    result.setClassifier(m_Classifier);
    result.setNumInstances(20);
    result.setDebug(DEBUG);
    result.setPostProcessor(getPostProcessor());

    return result;
  }

  /**
   * Configures the CheckGOE used for testing GOE stuff.
   * 
   * @return	the fully configured CheckGOE
   */
  @Override
  protected CheckGOE getGOETester() {
    CheckGOE		result;
    
    result = super.getGOETester();
    result.setIgnoredProperties(result.getIgnoredProperties() + ",testSet");
    
    return result;
  }

  /**
   * Builds a model using the current classifier using the first
   * half of the current data for training, and generates a bunch of
   * predictions using the remaining half of the data for testing.
   *
   * @param data 	the instances to test the classifier on
   * @return a <code>FastVector</code> containing the predictions.
   */
  @Override
  protected ArrayList<Prediction> useClassifier(Instances data) throws Exception {
    Classifier dc = null;
    int tot = data.numInstances();
    int mid = tot / 2;
    Instances train = null;
    Instances test = null;
    EvaluationUtils evaluation = new CollectiveEvaluationUtils();
    
    try {
      train = new Instances(data, 0, mid);
      test = new Instances(data, mid, tot - mid);
      dc = m_Classifier;
    } 
    catch (Exception e) {
      e.printStackTrace();
      fail("Problem setting up to use classifier: " + e);
    }

    do {
      try {
	return evaluation.getTrainTestPredictions(dc, train, test);
      } 
      catch (IllegalArgumentException e) {
	String msg = e.getMessage();
	if (msg.indexOf("Not enough instances") != -1) {
	  System.err.println("\nInflating training data.");
	  Instances trainNew = new Instances(train);
	  for (int i = 0; i < train.numInstances(); i++) {
	    trainNew.add(train.instance(i));
	  }
	  train = trainNew;
	} 
	else {
	  throw e;
	}
      }
    } while (true);
  }

  /**
   * Disabled for the time being.
   */
  @Override
  public void testBuildInitialization() {
  }

  /**
   * Disabled for the time being.
   */
  @Override
  public void testMissingPredictors() {
  }

  /**
   * Disabled for the time being.
   */
  @Override
  public void testAttributes() {
  }

  /**
   * Disabled for the time being.
   */
  @Override
  public void testUseOfTestClassValue() {
  }
}
