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
 * CollectiveInstances.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;

import java.io.Serializable;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.Tag;
import weka.core.Utils;

/**
 * A helper class for initializing labels and flipping class labels.
 * 
 * @author  Fracpete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $ 
 */
public class CollectiveInstances 
  implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 3867860167684511888L;

  /** we perform a random walk and use the last model for prediction */
  public final static int EVAL_RANDOMWALK_LAST = 0;

  /** we perform a random walk, but use the best model encountered for
   * prediction */
  public final static int EVAL_RANDOMWALK_BEST = 1;

  /** we perform sort of hillclimbing by always using the best model */
  public final static int EVAL_HILLCLIMBING = 2;

  /** the tags to select in the GUI */
  public static final Tag[] EVAL_TAGS = {
    new Tag(EVAL_RANDOMWALK_LAST, "Randomwalk, use last model for pred."),
    new Tag(EVAL_RANDOMWALK_BEST, "Randomwalk, use best model for pred."),
    new Tag(EVAL_HILLCLIMBING,    "Hillclimbing")
  };

  /** the RMS is used for comparing models */
  public final static int COMPARE_RMS = 0;

  /** the RMS on the training set is used for comparing models */
  public final static int COMPARE_RMS_TRAIN = 1;

  /** the RMS on the test set is used for comparing models */
  public final static int COMPARE_RMS_TEST = 2;

  /** the accuracy on the training set is used for comparing models */
  public final static int COMPARE_ACC_TRAIN = 3;

  /** the tags to select in the GUI */
  public static final Tag[] COMPARE_TAGS = {
    new Tag(COMPARE_RMS,       "Use overall RMS for comparing models"),
    new Tag(COMPARE_RMS_TRAIN, "Use RMS on training set for comparing models"),
    new Tag(COMPARE_RMS_TEST,  "Use RMS on test set for comparing models"),
    new Tag(COMPARE_ACC_TRAIN, "Use Accuracy on training set for comparing models")
  };
  
  /** for debugging information */
  protected boolean m_Debug = false;
  
  /** for random number generation */
  protected Random m_Random = null;
  
  /** the seed value for random numbers (default: 1) */
  protected int m_Seed = 1;

  /** the percentage of labels that were flipped */
  protected double m_FlippedLabels = 0;

  /** the flipping algorithm to use */
  protected Flipper m_Flipper = new TriangleFlipper();
  
  /** 
   * Creates a new instance of CollectiveInstances 
   */
  public CollectiveInstances() {
    super();

    setSeed(m_Seed);
  }
  
  /**
   * sets the seed value for random number generation
   * 
   * @param seed	the seed value
   */
  public void setSeed(int seed) {
    m_Seed = seed;
    m_Random = new Random(m_Seed);
  }
  
  /**
   * returns the seed value for random number generation
   * 
   * @return		the current seed value
   */
  public int getSeed() {
    return m_Seed;
  }
  
  /**
   * Sets debugging mode.
   *
   * @param debug true if debug output should be printed
   */
  public void setDebug(boolean debug) {
    m_Debug = debug;
  }
  
  /**
   * Gets whether debugging is turned on.
   *
   * @return true if debugging output is on
   */
  public boolean getDebug() {
    return m_Debug;
  }

  /**
   * sets the flipping algorithm
   * @param f       the new flipping algorithm
   */
  public void setFlipper(Flipper f) {
    m_Flipper = f;
  }

  /**
   * returns the current flipping algorithm
   * 
   * @return		the current flipper
   */
  public Flipper getFlipper() {
    return m_Flipper;
  }

  /**
   * returns the percentage of labels that were flipped in the flipLabels
   * method
   * @return      the percentage of flipped labels
   * @see         #flipLabels(Classifier,Instances,FlipHistory)
   */
  public double getFlippedLabels() {
    return m_FlippedLabels;
  }

  /**
   * randomly initializes the class labels in the given set according to the
   * class distribution in the training set
   * @param train       the training instances to retrieve the class
   *                    distribution from
   * @param instances   the instances to initialize
   * @return            the initialize instances
   * @throws Exception  if something goes wrong
   */
  public Instances initializeLabels(Instances train, Instances instances) 
    throws Exception {

    return initializeLabels(train, instances, 0, instances.numInstances());
  }

  /**
   * randomly initializes the class labels in the given set according to the
   * class distribution in the training set
   * @param train       the training instances to retrieve the class
   *                    distribution from
   * @param instances   the instances to initialize
   * @param from        the first instance to initialize
   * @param count       the number of instances to initialize
   * @return            the initialize instances
   * @throws Exception  if something goes wrong
   */
  public Instances initializeLabels( Instances train, Instances instances, 
                                     int from, int count )
    throws Exception {
      
    int             i;
    AttributeStats  stats;
    Attribute       classAttr;
    double          percentage;
    
    // reset flip count
    m_FlippedLabels = 0;
    
    // explicitly set labels to "missing"
    for (i = from; i < from + count; i++)
      instances.instance(i).setClassMissing();
    
    // determining the percentage of the first class
    stats      = train.attributeStats(train.classIndex());
    percentage = (double) stats.nominalCounts[0] / (double) stats.totalCount;
    
    // set lables
    classAttr = instances.attribute(instances.classIndex());
    for (i = from; i < from + count; i++) {
      // random class
      if (m_Random.nextDouble() < percentage)
        instances.instance(i).setClassValue(classAttr.value(0));
      else
        instances.instance(i).setClassValue(classAttr.value(1));
    }

    return instances;
  }

  /**
   * flips labels in the part of the given set where the probability is close
   * to 0.5
   * @param c             the current collective classifier
   * @param instances     the instances to work on
   * @param history       the flipping history
   * @return              the flipped instances
   * @throws Exception    if something goes wrong
   */
  public Instances flipLabels( Classifier c, Instances instances, 
                               FlipHistory history ) 
    throws Exception {

    return flipLabels(c, instances, 0, instances.numInstances(), history);
  }

  /**
   * flips labels in the part of the given set, with the current flipping'
   * algorithm. 
   * @param c             the current collective classifier
   * @param instances     the instances to work on
   * @param from          the first instance to flip
   * @param count         the number of instances to flip
   * @param history       the flipping history
   * @return              the flipped instances
   * @throws Exception    if something goes wrong
   * @see                 #setFlipper(Flipper)
   */
  public Instances flipLabels( Classifier c, Instances instances,
                               int from, int count, FlipHistory history ) 
    throws Exception {

    int         i;
    double      oldLabel;
    double      newLabel;
    
    // reset flip count
    m_FlippedLabels = 0;
    
    m_Flipper.setRandom(m_Random);

    for (i = from; i < from + count; i++) {
      oldLabel = instances.instance(i).classValue();
      newLabel = m_Flipper.flipLabel(c, instances, from, count, i, history);

      instances.instance(i).setClassValue(newLabel);
      
      // keep track of flipped labels
      if (!Utils.eq(oldLabel, newLabel))
        m_FlippedLabels += 1.0 / count;
    }

    return instances;
  }

  /**
   * calculates the RMS for test/original test and train set 
   * @param c             the classifier to use for determining the RMS
   * @param train         the training set
   * @param test          the test set
   * @param testOriginal  the original test set (can be null)
   * @return              the RMS array (contains RMS, RMSTrain, RMSTest, 
   *                      RMSTestOriginal)
   * @throws Exception    if something goes wrong
   */
  public static double[] calculateRMS( Classifier c, 
                                       Instances train, 
                                       Instances test,
                                       Instances testOriginal )
    throws Exception {

    int         i;
    double[]    dist;
    double[]    result;
    
    result = new double[4];
    
    // 1. train
    result[1] = 0;
    for (i = 0; i < train.numInstances(); i++) {
      dist = c.distributionForInstance(train.instance(i));
      result[1] += StrictMath.pow(
                      dist[StrictMath.abs((int) 
                         train.instance(i).classValue() - 1)], 2);
    }

    // 2. test
    result[2] = 0;
    for (i = 0; i < test.numInstances(); i++) {
      dist = c.distributionForInstance(test.instance(i));
      result[2] += StrictMath.pow(StrictMath.min(dist[0], dist[1]), 2);
    }
    
    // 4. original test
    if (testOriginal != null) {
      result[3] = 0;
      for (i = 0; i < testOriginal.numInstances(); i++) {
        dist = c.distributionForInstance(testOriginal.instance(i));
        result[3] += StrictMath.pow(
                        dist[StrictMath.abs((int) 
                           testOriginal.instance(i).classValue() - 1)], 2);
      }
    }
    else {
      result[3] = Double.NaN;
    }
    
    // normalize
    result[0] = (result[2] + result[1]) / 
                    (test.numInstances() + train.numInstances());
    result[1] = result[1] / train.numInstances();
    result[2] = result[2] / test.numInstances();
    if (testOriginal != null)
      result[3] = result[3] / testOriginal.numInstances();

    // root
    result[0] = StrictMath.sqrt(result[0]);
    result[1] = StrictMath.sqrt(result[1]);
    result[2] = StrictMath.sqrt(result[2]);
    if (testOriginal != null)
      result[3] = StrictMath.sqrt(result[3]);

    return result;
  }

  /**
   * calculates the accuracy for original test and train set 
   * @param c		the classifier to use for determining the RMS
   * @param train	the training set
   * @param test	the original test set
   * @return		the accuracy array (contains AccTrain, AccTestOriginal)
   * @throws Exception 	if something goes wrong
   */
  public static double[] calculateAccuracy( Classifier c, 
                                            Instances train, 
                                            Instances test )
    throws Exception {

    int         i;
    double      classValue;
    double[]    result;
    
    result = new double[2];
    
    // 1. train
    result[0] = 0;
    for (i = 0; i < train.numInstances(); i++) {
      classValue = c.classifyInstance(train.instance(i));
      if (Utils.eq(classValue, train.instance(i).classValue()))
        result[0] += 1.0;
    }

    // 2. test
    result[1] = 0;
    if (test != null) {
      for (i = 0; i < test.numInstances(); i++) {
        classValue = c.classifyInstance(test.instance(i));
        if (Utils.eq(classValue, test.instance(i).classValue()))
          result[1] += 1.0;
      }
    }
    
    // normalize
    result[0] = result[0] / train.numInstances();
    if (test != null)
      result[1] = result[1] / test.numInstances();

    return result;
  }
}
