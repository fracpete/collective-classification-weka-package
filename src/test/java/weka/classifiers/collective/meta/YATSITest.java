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

package weka.classifiers.collective.meta;

import junit.framework.Test;
import junit.framework.TestSuite;
import weka.classifiers.Classifier;
import weka.classifiers.collective.AbstractCollectiveClassifierTest;

/**
 * Tests YATSI. Run from the command line with:<p>
 * java weka.classifiers.collective.meta.YATSITest
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $
 */
public class YATSITest 
  extends AbstractCollectiveClassifierTest {

  public YATSITest(String name) { 
    super(name);  
  }

  /** Creates a default YATSI */
  @Override
  public Classifier getClassifier() {
    return new YATSI();
  }

  /**
   * Disabled for the time being.
   */
  @Override
  public void testZeroTraining() {
  }

  public static Test suite() {
    return new TestSuite(YATSITest.class);
  }

  public static void main(String[] args){
    junit.textui.TestRunner.run(suite());
  }
}
