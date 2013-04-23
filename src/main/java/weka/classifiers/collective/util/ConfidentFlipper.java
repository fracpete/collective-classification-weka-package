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
 * ConfidentFlipper.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
  
/**
 <!-- globalinfo-start -->
 * Flips a label only if the previous and the current prediction differ more than a given percentage. The flipping is then done randomly.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -delta &lt;num&gt;
 *  The minimum percentage for disagreement to enable flipping.
 *  (default is 0.75)</pre>
 * 
 * <pre> -D
 *  Turns on output of debugging information.
 *  (default is off)</pre>
 * 
 <!-- options-end -->
 *
 * @author    Kurt Driessens (kurtd at cs dot waikato dot ac dot nz)
 * @author    FracPete (fracpete at waikato dot ac dot nz)
 * @version   $Revision: 2019 $
 */
public class ConfidentFlipper 
  extends Flipper {

  /** for serialization */
  private static final long serialVersionUID = -3915210066408087192L;
  
  /** the minimum level of disagreement */
  protected double m_Delta = 0.75;

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  @Override
  public String globalInfo() {
    return   "Flips a label only if the previous and the current prediction "
           + "differ more than a given percentage. The flipping is then done "
           + "randomly.";
  }

  /**
   * Returns an enumeration of all the available options..
   *
   * @return an enumeration of all available options.
   */
  @Override
  public Enumeration listOptions() {
    Vector result = new Vector();
    
    result.addElement(new Option(
        "\tThe minimum percentage for disagreement to enable flipping.\n"
        + "\t(default is 0.75)",
        "delta", 1, "-delta <num>"));
    
    Enumeration en = super.listOptions();
    while (en.hasMoreElements())
      result.addElement(en.nextElement());
      
    return result.elements();
  }

  /**
   * Sets the OptionHandler's options using the given list. All options
   * will be set (or reset) during this call (i.e., incremental setting
   * of options is not possible). <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -delta &lt;num&gt;
   *  The minimum percentage for disagreement to enable flipping.
   *  (default is 0.75)</pre>
   * 
   * <pre> -D
   *  Turns on output of debugging information.
   *  (default is off)</pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {
    String        tmpStr;
    
    tmpStr = Utils.getOption("delta", options);
    if (tmpStr.length() != 0)
      setDelta(Double.parseDouble(tmpStr));
    else
      setDelta(0.75);
    
    super.setOptions(options);
  }

  /**
   * Gets the current option settings for the OptionHandler.
   *
   * @return the list of current option settings as an array of strings
   */
  @Override
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;
    
    result  = new Vector();

    result.add("-delta");
    result.add("" + getDelta());
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    return (String[]) result.toArray(new String[result.size()]);
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String deltaTipText() {
    return "The minimum percentage of disagreement between previous and current prediction that needs to be met in order to enable flipping.";
  }
  
  /**
   * Sets the minimum disagreement to enable flipping
   * 
   * @param value	the delta to use
   */
  public void setDelta(double value) {
    if ( (value > 0) && (value < 1) )
      m_Delta = value;
    else
      System.out.println("Must be between 0 and 1 (provided: " + value + ")!");
  }
  
  /**
   * Gets the minimum disagreement to enable flipping
   *
   * @return the minimum disagreement
   */
  public double getDelta() {
    return m_Delta;
  }

  /**
   * returns the (possibly) new class label
   * @param c           the Classifier to use for prediction
   * @param instances   the instances to use for flipping
   * @param from        the starting of flipping
   * @param count       the number of instances to flip
   * @param index       the index of the instance to flip
   * @param history	the flipping history
   * @return            the (possibly) new class label
   */
  @Override
  public double flipLabel( Classifier c, Instances instances, 
                           int from, int count, int index,
                           FlipHistory history ) {
    double[]        dist;
    double          result;
    
    // get distribution 
    try {
      dist = c.distributionForInstance(instances.instance(index));
    }
    catch (Exception e) {
      e.printStackTrace();
      return instances.instance(index).classValue();
    }
    
    // do we disagree enough?
    if ( StrictMath.abs(
            dist[0] - history.getLast(instances.instance(index))[0]) 
         >= getDelta()) {
      // flip label
      if (m_Random.nextDouble() < dist[0])
        result = dist[0];
      else
        result = dist[1];
    }
    else {
      result = (instances.instance(index).classValue());
    }

    // history
    history.add(instances.instance(index), dist);

    return result;
  }
}
