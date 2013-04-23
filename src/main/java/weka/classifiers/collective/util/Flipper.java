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
 * Flipper.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
  
/**
 * Abstract class for algorithms that flip labels according to a certain 
 * function. <p/>
 *
 * @author    FracPete (fracpete at waikato dot ac dot nz)
 * @version   $Revision: 2019 $
 */
public abstract class Flipper 
  implements Serializable, OptionHandler {
  
  /** for debugging information */
  protected boolean m_Debug = false;

  /** the random number generator to use */
  protected Random m_Random = null;

  /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public abstract String globalInfo();
  
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
   * sets the random number generator to use
   * 
   * @param r		the random number generator to use
   */
  public void setRandom(Random r) {
    m_Random = r;
  }

  /**
   * returns the Random number generator that is used
   * 
   * @return		the current number generator
   */
  public Random getRandom() {
    return m_Random;
  }

  /**
   * Returns an enumeration of all the available options..
   *
   * @return an enumeration of all available options.
   */
  public Enumeration listOptions() {
    Vector result = new Vector();
    
    result.addElement(new Option(
        "\tTurns on output of debugging information.\n"
        + "\t(default is off)",
        "D", 0, "-D"));
    
    return result.elements();
  }

  /**
   * Sets the OptionHandler's options using the given list. All options
   * will be set (or reset) during this call (i.e., incremental setting
   * of options is not possible).
   *
   * Available options: <p/>
   *
   * -D <br/>
   *  turns on output of debugging information <p/>
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    setDebug(Utils.getFlag('D', options));
  }

  /**
   * Gets the current option settings for the OptionHandler.
   *
   * @return the list of current option settings as an array of strings
   */
  public String[] getOptions() {
    Vector        result;
    
    result = new Vector();

    if (getDebug())
      result.add("-D");
    
    return (String[]) result.toArray(new String[result.size()]);
  }

  /**
   * returns the (possibly) new class label. The Random number generator
   * must be set before!
   * @param c           the Classifier to use for prediction
   * @param instances   the instances to use for flipping
   * @param from        the starting of flipping
   * @param count       the number of instances to flip
   * @param index       the index of the instance to flip
   * @param history     the flipping history
   * @return            the (possibly) new class label
   * @see               #setRandom(Random)
   */
  public abstract double flipLabel( Classifier c, Instances instances, 
                           int from, int count, int index,
                           FlipHistory history );

  /**
   * sets all the options from the given flipping algorithm
   * @param f     the flipper to get the options from
   */
  public void assign(Flipper f) {
    setDebug(f.getDebug());
    setRandom(f.getRandom());
  }
  
  /**
   * returns the specification of the given optionhandler (class + options)
   *
   * @param o		the option handler to get the specs as string
   * @return		the specification string
   */
  public static String getSpecification(OptionHandler o) {
     return o.getClass().getName() + " " + Utils.joinOptions(o.getOptions());
  }

  /**
   * generates a new flip algorithm instance, based on the classname and
   * the given options
   * 
   * @param classname   the classname of the flip algorithm to create
   * @param options     (optional) options for the algorithm
   * @return            the flipping algorithm
   * @throws Exception  if creation fails
   */
  public static Flipper forName(String classname, String[] options) 
    throws Exception {

    Flipper       result;

    result = (Flipper) Class.forName(classname).newInstance();
    result.setOptions(options);

    return result;
  }
}
