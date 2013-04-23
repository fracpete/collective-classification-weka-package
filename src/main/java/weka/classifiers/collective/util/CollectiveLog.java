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
 * CollectiveLog.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Hashtable;
  
/**
 * Used to output some internal statistics for a Collective Classifier.
 * It's used to generate CSV files, i.e., the addValue method adds new
 * columns, until they're written with the write method to the file as
 * a new row.
 *
 * @author    FracPete (fracpete at waikato dot ac dot nz)
 * @version   $Revision: 2019 $
 */
public class CollectiveLog
  implements Serializable {
  
  /** for serialization */
  private static final long serialVersionUID = 6745713386327448747L;

  /** keeps track of the value-filename relation */
  private Hashtable m_Filenames;
  
  /** keeps track of the values */
  private Hashtable m_Values;
  
  /**
   * initializes the log
   */
  public CollectiveLog() {
    clear();
  }

  /**
   * removes all the values and filenames
   */
  public void clear() {
    m_Filenames = new Hashtable();
    m_Values    = new Hashtable();
  }

  /**
   * adds the identifier-filename relation
   * @param identifier      the identifier to set the filename for
   * @param filename        the filename to use as output for the values of
   *                        the identifier
   */
  public void addFilename(String identifier, String filename) {
    m_Filenames.put(identifier, filename);
  }

  /**
   * adds the given value to the currently stored values for the given 
   * identifier
   * @param identifier    the identifier to add the value for
   * @param value         the value to add
   */
  public void addValue(String identifier, int value) {
    addValue(identifier, "" + value);
  }

  /**
   * adds the given value to the currently stored values for the given 
   * identifier
   * @param identifier    the identifier to add the value for
   * @param value         the value to add
   */
  public void addValue(String identifier, double value) {
    addValue(identifier, "" + value);
  }

  /**
   * adds the given value to the currently stored values for the given 
   * identifier
   * @param identifier    the identifier to add the value for
   * @param value         the value to add
   */
  public void addValue(String identifier, String value) {
    String        values;

    // retrieve, if possible
    values = getValues(identifier);

    // add value
    if (!values.equals(""))
      values += ",";
    values += value;

    // add again
    setValues(identifier, values);
  }

  /**
   * returns the values for the given identifier
   * @param identifier    the identifier to retrieve the values for
   * @return              the associated values
   */
  public String getValues(String identifier) {
    String      result;

    if (m_Values.containsKey(identifier))
      result = (String) m_Values.get(identifier);
    else
      result = "";

    return result;
  }

  /**
   * stores the given values under the specified identifier
   * @param identifier    the identifier the values are associated with
   * @param values        the values to store
   */
  public void setValues(String identifier, String values) {
    m_Values.put(identifier, values);
  }

  /**
   * returns the filename for the given identifier
   * 
   * @param identifier	the id to retrieve the filename for
   * @return		the filename, if one was found, otherwise an
   * 			empty string
   */
  public String getFilename(String identifier) {
    String      result;

    if (m_Filenames.containsKey(identifier))
      result = (String) m_Filenames.get(identifier);
    else
      result = "";

    return result;
  }

  /**
   * writes all the currently stored values to the associated files. The
   * values are then reset, i.e., set to the empty string.
   */
  public void write() {
    Enumeration     enm;
    String          key;

    enm = m_Filenames.keys();

    while (enm.hasMoreElements()) {
      key = (String) enm.nextElement();
      CollectiveHelper.writeToTempFile(getFilename(key), getValues(key), true);
      setValues(key, "");
    }
  }
}
