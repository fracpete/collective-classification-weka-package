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
 * TriangleFlipper.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Utils;
  
/**
 <!-- globalinfo-start -->
 * flip new label if (rnd &lt; max(5/#instances, 1-2*(prob-0.5)))<br/>
 * with prob as the probability the classifier predicted
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D
 *  Turns on output of debugging information.
 *  (default is off)</pre>
 * 
 <!-- options-end -->
 *
 * @author    Bernhard Pfahringer (bernhard at cs dot waikato dot ac dot nz)
 * @author    FracPete (fracpete at waikato dot ac dot nz)
 * @version   $Revision: 2019 $
 */
public class TriangleFlipper 
  extends Flipper {

  /** for serialization */
  private static final long serialVersionUID = 3458916344540907433L;

  /**
   * Returns a string describing classifier
   * 
   * @return 		a description suitable for
   * 			displaying in the explorer/experimenter gui
   */
  @Override
  public String globalInfo() {
    return   "flip new label if (rnd < max(5/#instances, 1-2*(prob-0.5)))\n"
           + "with prob as the probability the classifier predicted";
  }

  /**
   * returns the (possibly) new class label
   * 
   * @param c           the Classifier to use for prediction
   * @param instances   the instances to use for flipping
   * @param from        the starting of flipping
   * @param count       the number of instances to flip
   * @param index       the index of the instance to flip
   * @param history     the flipping history
   * @return            the (possibly) new class label
   */
  @Override
  public double flipLabel( Classifier c, Instances instances, 
                           int from, int count, int index,
                           FlipHistory history ) {
    double          result;
    double[]        dist;
    double          prob;
    double          rand;
    double          threshold;
    
    try {
      result = c.classifyInstance(instances.instance(index));
      dist   = c.distributionForInstance(instances.instance(index));
    } 
    catch (Exception e) {
      e.printStackTrace();
      return instances.instance(index).classValue();
    }
    
    prob = dist[(int) result];
    
    // flip label
    rand      = m_Random.nextDouble();
    threshold = StrictMath.max(5.0 / count, 1 - 2*(prob - 0.5));
    if (rand < threshold) {
      if (Utils.eq(result, 0.0))
        result = 1;
      else
        result = 0;
    }

    // history
    history.add(instances.instance(index), dist);

    return result;
  }
}
