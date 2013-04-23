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
 * Clock.java
 * Copyright (C) 2005-2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.collective.util;

import java.io.Serializable;

import weka.core.Utils;

/**
 * A little helper class for clocking and outputting times.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision: 2019 $ 
 */
public class Clock 
  implements Serializable {

  /** for serialization */
  private static final long serialVersionUID = 4622161807307942201L;

  /** the output format in seconds, with fraction of msecs */
  public final static int FORMAT_SECONDS = 0;

  /** the output format in hours:minutes:seconds, with fraction of msecs */
  public final static int FORMAT_HHMMSS = 1;

  /** the format of the output, either FORMAT_SECONDS or FORMAT_HHMMSS */
  public int m_OutputFormat = FORMAT_SECONDS;
  
  /** the start time */
  protected long m_Start;

  /** the end time */
  protected long m_Stop;

  /** whether the time is still clocked */
  protected boolean m_Running;
  
  /**
   * automatically starts the clock with FORMAT_SECONDS format
   * 
   * @see               #m_OutputFormat
   */
  public Clock() {
    this(true);
  }

  /**
   * automatically starts the clock with the given output format
   * 
   * @param format      the output format
   * @see               #m_OutputFormat
   */
  public Clock(int format) {
    this(true, format);
  }

  /**
   * starts the clock depending on <code>start</code> immediately with the
   * FORMAT_SECONDS output format
   * 
   * @param start       whether to start the clock immediately
   * @see               #m_OutputFormat
   */
  public Clock(boolean start) {
    this(start, FORMAT_SECONDS);
  }

  /**
   * starts the clock depending on <code>start</code> immediately
   * 
   * @param start       whether to start the clock immediately
   * @param format	the format
   * @see               #m_OutputFormat
   */
  public Clock(boolean start, int format) {
    m_Running = false;
    m_Start   = 0;
    m_Stop    = 0;
    setOutputFormat(format);
    
    if (start)
      start();
  }

  /**
   * saves the current system time in msec as start time
   * 
   * @see       #m_Start
   */
  public void start() {
    m_Start   = System.currentTimeMillis();
    m_Stop    = m_Start;
    m_Running = true;
  }

  /**
   * saves the current system in msec as stop time
   * 
   * @see       #m_Stop
   */
  public void stop() {
    m_Stop    = System.currentTimeMillis();
    m_Running = false;
  }

  /**
   * returns the start time
   * 
   * @return	the start time
   */
  public long getStart() {
    return m_Start;
  }

  /**
   * returns the stop time or, if still running, the current time
   * 
   * @return 	the stop time
   */
  public long getStop() {
    if (isRunning())
      return System.currentTimeMillis();
    else
      return m_Stop;
  }

  /**
   * whether the time is still being clocked
   * 
   * @return		true if the time is still being clocked
   */
  public boolean isRunning() {
    return m_Running;
  }

  /**
   * sets the format of the output
   * 
   * @param value       the format of the output
   * @see               #m_OutputFormat
   */
  public void setOutputFormat(int value) {
    if (value == FORMAT_SECONDS)
      m_OutputFormat = value;
    else if (value == FORMAT_HHMMSS)
      m_OutputFormat = value;
    else
      System.out.println("Format '" + value + "' is not recognized!");
  }

  /**
   * returns the output format
   * 
   * @return		the output format
   * @see		#m_OutputFormat
   */
  public int getOutputFormat() {
    return m_OutputFormat;
  }

  /**
   * returns the elapsed time, getStop() - getStart(), as string
   * 
   * @return	the elapsed time as string
   * @see       #getStart()
   * @see       #getStop()
   */
  @Override
  public String toString() {
    String    result;
    long      elapsed;
    long      hours;
    long      mins;
    long      secs;
    long      msecs;

    result  = "";
    elapsed = getStop() - getStart();

    switch (getOutputFormat()) {
      case FORMAT_HHMMSS:
        hours   = elapsed / (3600 * 1000);
        elapsed = elapsed % (3600 * 1000);
        mins    = elapsed / (60 * 1000);
        elapsed = elapsed % (60 * 1000);
        secs    = elapsed / 1000;
        msecs   = elapsed % 1000;

        if (hours > 0)
          result += "" + hours + ":";
        
        if (mins < 10)
          result += "0" + mins + ":";
        else
          result += ""  + mins + ":";
        
        if (secs < 10)
          result += "0" + secs + ".";
        else
          result += "" + secs + ".";
        
        result += Utils.doubleToString(
                    (double) msecs / (double) 1000, 3).replaceAll(".*\\.", "");
        break;

      case FORMAT_SECONDS:
        result = Utils.doubleToString((double) elapsed / (double) 1000, 3);
        break;
    }
    
    return result;
  }

  /**
   * for testing only
   * 
   * @param args	the commandline arguments - ignored
   */
  public static void main(String[] args) {
    Clock     c;

    c = new Clock();
    System.out.println(c);
    double[] d = new double[1000000];
    System.out.println(c);
    System.gc();
    System.out.println(c);
    c.stop();
    System.out.println(c);
    double[] e = new double[1000000];
    System.out.println(c);
    c.setOutputFormat(FORMAT_HHMMSS);
    System.out.println(c);
    System.gc();
  }
}
